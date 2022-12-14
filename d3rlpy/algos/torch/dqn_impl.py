import copy
from typing import Optional, Sequence, List, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from ...preprocessing import RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api, l2_regularized_loss
from .base import TorchImplBase
from .utility import DiscreteQFunctionMixin
from .utility import disable_running_stats, enable_running_stats
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density


class DQNImpl(DiscreteQFunctionMixin, TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None
        ######## For SAM ##########
        if 'SAM' in self._optim_factory._optim_cls.__name__:
            def closure():
                self._optim.zero_grad()
                #q_tpn = self.compute_target(batch)
                #loss = self._compute_critic_loss(batch, q_tpn)
                loss = self.compute_critic_loss(batch)
                loss.backward()
                return loss
        else:
            closure = None
        ###########################

        self._optim.zero_grad()

        #q_tpn = self.compute_target(batch)
        #loss = self._compute_critic_loss(batch, q_tpn)
        loss = self.compute_critic_loss(batch)

        loss.backward()
        #self._optim.step()
        ######## For SAM ##########
        loss_sam = self._optim.step(closure)
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
            return l2_regularized_loss(loss, self._q_func, self._optim)
        return loss

    def _compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._optim
        return self._optim
    
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


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        with torch.no_grad():
            action = self._predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
