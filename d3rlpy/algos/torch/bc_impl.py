from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union, List, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_deterministic_policy,
    create_deterministic_regressor,
    create_discrete_imitator,
    create_probablistic_regressor,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import (
    DeterministicRegressor,
    DiscreteImitator,
    Imitator,
    Policy,
    ProbablisticRegressor,
)
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api, l2_regularized_loss
from .base import TorchImplBase
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density


class BCBaseImpl(TorchImplBase, metaclass=ABCMeta):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _imitator: Optional[Imitator]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=None,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._use_gpu = use_gpu

        # initialized in build
        self._imitator = None
        self._optim = None

    def build(self) -> None:
        self._build_network()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._imitator is not None
        self._optim = self._optim_factory.create(
            self._imitator.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_imitator(
        self, batch: TorchMiniBatch
    ) -> np.ndarray:
        assert self._optim is not None
        ######## For SAM ##########
        if 'SAM' in self._optim_factory._optim_cls.__name__:
            def closure():
                self._optim.zero_grad()
                loss = self.compute_imitator_loss(batch)
                loss.backward()
                return loss
        else:
            closure = None
        ###########################

        self._optim.zero_grad()

        loss = self.compute_imitator_loss(batch)

        loss.backward()
        #self._optim.step()
        ######## For SAM ##########
        loss_sam = self._optim.step(closure)
        if loss_sam is not None:
            loss_sharpness = loss_sam.cpu().detach().numpy() - loss.cpu().detach().numpy()
            return loss.cpu().detach().numpy(), loss_sharpness  # sharpness added!
        ###########################

        return loss.cpu().detach().numpy()
    
    def compute_imitator_loss(
        self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False,
    ) -> torch.Tensor:
        assert self._imitator is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        loss = self._imitator.compute_error(batch.observations, batch.actions)
        if l2_reg:
            return l2_regularized_loss(loss, self._imitator, self._optim)
        return loss

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x)

    def predict_value(
        self, x: np.ndarray, action: np.ndarray, with_std: bool
    ) -> np.ndarray:
        raise NotImplementedError("BC does not support value estimation")

    def hessian_eig_imitator(self,
        iterator: TransitionIterator,
        top_n: int,
        max_iter: int,
        tolerance: Optional[float],
        show_progress: Optional[bool],
    ) -> List[float]:
        return hessian_eigenvalues(self._imitator, self.compute_imitator_loss, iterator, top_n, max_iter, tolerance, show_progress, device=self.device)
    
    def hessian_spectra_imitator(self,
        iterator: TransitionIterator,
        n_run: int,
        max_iter: int,
        show_progress: Optional[bool]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        eigenvalues, weights = hessien_empirical_spectral_density(
            self._imitator, self.compute_imitator_loss, iterator, n_run, max_iter, show_progress, device=self.device
        )
        return eigenvalues, weights


class BCImpl(BCBaseImpl):

    _policy_type: str
    _imitator: Optional[Union[DeterministicRegressor, ProbablisticRegressor]]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        policy_type: str,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
        )
        self._policy_type = policy_type

    def _build_network(self) -> None:
        if self._policy_type == "deterministic":
            self._imitator = create_deterministic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            self._imitator = create_probablistic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
            )
        else:
            raise ValueError("invalid policy_type: {self._policy_type}")

    @property
    def policy(self) -> Policy:
        assert self._imitator

        policy: Policy
        if self._policy_type == "deterministic":
            policy = create_deterministic_policy(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            policy = create_squashed_normal_policy(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
                min_logstd=-20.0,
                max_logstd=2.0,
            )
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")

        # copy parameters
        hard_sync(policy, self._imitator)

        return policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._optim
        return self._optim


class DiscreteBCImpl(BCBaseImpl):

    _beta: float
    _imitator: Optional[DiscreteImitator]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=None,
        )
        self._beta = beta

    def _build_network(self) -> None:
        self._imitator = create_discrete_imitator(
            self._observation_shape,
            self._action_size,
            self._beta,
            self._encoder_factory,
        )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x).argmax(dim=1)

    def compute_imitator_loss(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        assert self._imitator is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        return self._imitator.compute_error(batch.observations, batch.actions.long())