from typing import Optional, Sequence, List, Tuple

import numpy as np
import torch

from ...gpu import Device
from ...models.builders import (
    create_non_squashed_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import MeanQFunctionFactory
from ...models.torch import NonSquashedNormalPolicy, ValueFunction
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api, l2_regularized_loss
from .ddpg_impl import DDPGBaseImpl
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density


class IQLImpl(DDPGBaseImpl):
    _policy: Optional[NonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

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
        value_encoder_factory: EncoderFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
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
            q_func_factory=MeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_value_function(
            self._observation_shape, self._value_encoder_factory
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

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

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        with torch.no_grad():
            return self._value_func(batch.next_observations)

    def compute_actor_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        assert self._policy
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        loss = -(weight * log_probs).mean()
        if l2_reg:
            return l2_regularized_loss(loss, self._policy, self._actor_optim)
        return loss 

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        loss = (weight * (diff**2)).mean()
        if l2_reg:
            return l2_regularized_loss(loss, self._value_func, self._critic_optim)
        return loss
    
    # compute_critic_loss = _compute_critic_loss + compute_value_loss
    def compute_critic_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        q_tpn = self.compute_target(batch)
        q_loss = self._compute_critic_loss(batch, q_tpn)
        v_loss = self.compute_value_loss(batch)
        loss = q_loss + v_loss
        if l2_reg:
            return l2_regularized_loss(loss, self._q_func, self._critic_optim)
        return loss

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None
        ######## For SAM ##########
        if 'SAM' in self._critic_optim_factory._optim_cls.__name__:
            def closure():
                q_tpn = self.compute_target(batch)
                q_loss = self._compute_critic_loss(batch, q_tpn)
                v_loss = self.compute_value_loss(batch)
                loss = q_loss + v_loss
                loss.backward()
                return q_loss, v_loss
        else:
            closure = None
        ###########################

        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self._compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        #self._critic_optim.step()
        ######## For SAM ##########
        losses_sam = self._critic_optim.step(closure)
        if losses_sam is not None:
            q_loss_sam, v_loss_sam = losses_sam
            q_loss_sam, v_loss_sam = q_loss_sam.cpu().detach().numpy(), v_loss_sam.cpu().detach().numpy()
            q_loss_sharpness = q_loss_sam - q_loss.cpu().detach().numpy()
            v_loss_sharpness = q_loss_sam - q_loss.cpu().detach().numpy()
            return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy(), q_loss_sharpness, v_loss_sharpness # sharpness added!
        ###########################

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()
    
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
