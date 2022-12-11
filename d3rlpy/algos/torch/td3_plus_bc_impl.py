# pylint: disable=too-many-ancestors

from typing import Optional, Sequence, List, Tuple

import torch

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, l2_regularized_loss
from .td3_impl import TD3Impl
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density



class TD3PlusBCImpl(TD3Impl):

    _alpha: float

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
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

    def compute_actor_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        loss = lam * -q_t.mean() + ((batch.actions - action) ** 2).mean()
        if l2_reg:
            return l2_regularized_loss(loss, self._policy, self._actor_optim)
        return loss
    
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
