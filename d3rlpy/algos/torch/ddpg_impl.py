import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, List, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    DeterministicPolicy,
    EnsembleContinuousQFunction,
    EnsembleQFunction,
    Policy,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api, l2_regularized_loss
from .base import TorchImplBase
from .utility import ContinuousQFunctionMixin
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density


class DDPGBaseImpl(ContinuousQFunctionMixin, TorchImplBase, metaclass=ABCMeta):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleContinuousQFunction]
    _policy: Optional[Policy]
    _targ_q_func: Optional[EnsembleContinuousQFunction]
    _targ_policy: Optional[Policy]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._targ_policy = None
        self._actor_optim = None
        self._critic_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    @abstractmethod
    def _build_actor(self) -> None:
        pass

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None
        ######## For SAM ##########
        if 'SAM' in self._critic_optim_factory._optim_cls.__name__:
            def closure():
                self._critic_optim.zero_grad()
                #q_tpn = self.compute_target(batch)
                #loss = self._compute_critic_loss(batch, q_tpn)
                loss = self.compute_critic_loss(batch)
                loss.backward()
                return loss
        else:
            closure = None
        ###########################

        self._critic_optim.zero_grad()

        #q_tpn = self.compute_target(batch)
        #loss = self._compute_critic_loss(batch, q_tpn)
        loss = self.compute_critic_loss(batch)

        loss.backward()
        #self._critic_optim.step()
        ######## For SAM ##########
        loss_sam = self._critic_optim.step(closure)
        if loss_sam is not None:
            loss_sharpness = loss_sam.cpu().detach().numpy() - loss.cpu().detach().numpy()
            return loss.cpu().detach().numpy(), loss_sharpness  # sharpness added!
        ###########################

        return loss.cpu().detach().numpy()

    def compute_critic_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        q_tpn = self.compute_target(batch)
        loss = self._compute_critic_loss(batch, q_tpn)
        if l2_reg:
            return l2_regularized_loss(loss, self._q_func, self._critic_optim)
        return loss

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None
        ######## For SAM ##########
        if 'SAM' in self._actor_optim_factory._optim_cls.__name__:
            def closure():
                self._actor_optim.zero_grad()
                loss = self.compute_actor_loss(batch)
                loss.backward()
                return loss
        else:
            closure = None
        ###########################

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        #self._actor_optim.step()
        ######## For SAM ##########
        loss_sam = self._actor_optim.step(closure)
        if loss_sam is not None:
            loss_sharpness = loss_sam.cpu().detach().numpy() - loss.cpu().detach().numpy()
            return loss.cpu().detach().numpy(), loss_sharpness  # sharpness added!
        ###########################

        return loss.cpu().detach().numpy()

    @abstractmethod
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)

    def update_actor_target(self) -> None:
        assert self._policy is not None
        assert self._targ_policy is not None
        soft_sync(self._targ_policy, self._policy, self._tau)

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._actor_optim
        return self._actor_optim

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._critic_optim
        return self._critic_optim
    
    def hessian_eig_critic(self,
        iterator: TransitionIterator,
        top_n: int,
        max_iter: int,
        tolerance: Optional[float],
        show_progress: Optional[bool],
    ) -> List[float]:
        return hessian_eigenvalues(self._q_func, self.compute_critic_loss, iterator, top_n, max_iter, tolerance, show_progress, device=self.device)
    
    def hessian_spectra_critic(self,
        iterator: TransitionIterator,
        n_run: int,
        max_iter: int,
        show_progress: Optional[bool]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        eigenvalues, weights = hessien_empirical_spectral_density(
            self._q_func, self.compute_critic_loss, iterator, n_run, max_iter, show_progress, device=self.device
        )
        return eigenvalues, weights


class DDPGImpl(DDPGBaseImpl):

    _policy: Optional[DeterministicPolicy]
    _targ_policy: Optional[DeterministicPolicy]

    def _build_actor(self) -> None:
        self._policy = create_deterministic_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def compute_actor_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        loss = -q_t.mean()
        if l2_reg:
            return l2_regularized_loss(loss, self._policy, self._actor_optim)
        return loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)
    
    def hessian_eig_actor(self,
        iterator: TransitionIterator,
        top_n: int,
        max_iter: int,
        tolerance: Optional[float],
        show_progress: Optional[bool],
    ) -> List[float]:
        return hessian_eigenvalues(self._policy, self.compute_actor_loss, iterator, top_n, max_iter, tolerance, show_progress, device=self.device)
    
    def hessian_spectra_actor(self,
        iterator: TransitionIterator,
        n_run: int,
        max_iter: int,
        show_progress: Optional[bool]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        eigenvalues, weights = hessien_empirical_spectral_density(
            self._policy, self.compute_actor_loss, iterator, n_run, max_iter, show_progress, device=self.device
        )
        return eigenvalues, weights
