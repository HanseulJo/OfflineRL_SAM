from typing import Optional, Sequence, List, Tuple

import torch
import torch.nn.functional as F

from ...gpu import Device
from ...models.builders import create_non_squashed_normal_policy
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import NonSquashedNormalPolicy
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, l2_regularized_loss
from .ddpg_impl import DDPGBaseImpl
from ...iterators import TransitionIterator
from ...hessian_utils import hessian_eigenvalues, hessien_empirical_spectral_density


class CRRImpl(DDPGBaseImpl):

    _beta: float
    _n_action_samples: int
    _advantage_type: str
    _weight_type: str
    _max_weight: float
    _policy: Optional[NonSquashedNormalPolicy]
    _targ_policy: Optional[NonSquashedNormalPolicy]

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
        beta: float,
        n_action_samples: int,
        advantage_type: str,
        weight_type: str,
        max_weight: float,
        n_critics: int,
        tau: float,
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
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._beta = beta
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._max_weight = max_weight

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def compute_actor_loss(self, batch: TorchMiniBatch, l2_reg: Optional[bool] = False) -> torch.Tensor:
        assert self._policy is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        weight = self._compute_weight(batch.observations, batch.actions)
        loss = -(log_probs * weight).mean()
        if l2_reg:
            return l2_regularized_loss(loss, self._policy, self._actor_optim)
        return loss

    def _compute_weight(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        advantages = self._compute_advantage(obs_t, act_t)
        if self._weight_type == "binary":
            return (advantages > 0.0).float()
        elif self._weight_type == "exp":
            return (advantages / self._beta).exp().clamp(0.0, self._max_weight)
        raise ValueError(f"invalid weight type: {self._weight_type}.")

    def _compute_advantage(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert self._policy is not None
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # (batch_size, N, action)
            policy_actions = self._policy.sample_n(
                obs_t, self._n_action_samples
            )
            flat_actions = policy_actions.reshape(-1, self._action_size)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, 1, obs_size)
            reshaped_obs_t = obs_t.view(batch_size, 1, *obs_t.shape[1:])
            # (batch_sie, 1, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = reshaped_obs_t.expand(
                batch_size, self._n_action_samples, *obs_t.shape[1:]
            )
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = repeated_obs_t.reshape(-1, *obs_t.shape[1:])

            flat_values = self._q_func(flat_obs_t, flat_actions)
            reshaped_values = flat_values.view(obs_t.shape[0], -1, 1)

            if self._advantage_type == "mean":
                values = reshaped_values.mean(dim=1)
            elif self._advantage_type == "max":
                values = reshaped_values.max(dim=1).values
            else:
                raise ValueError(
                    f"invalid advantage type: {self._advantage_type}."
                )

            return self._q_func(obs_t, act_t) - values

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        if not isinstance(batch, TorchMiniBatch):
            batch = TorchMiniBatch(batch, self.device, self.scaler, self.action_scaler, self.reward_scaler)
        with torch.no_grad():
            action = self._targ_policy.sample(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None

        # compute CWP

        actions = self._policy.onnx_safe_sample_n(x, self._n_action_samples)
        # (batch_size, N, action_size) -> (batch_size * N, action_size)
        flat_actions = actions.reshape(-1, self._action_size)

        # repeat observation
        # (batch_size, obs_size) -> (batch_size, 1, obs_size)
        reshaped_obs_t = x.view(x.shape[0], 1, *x.shape[1:])
        # (batch_size, 1, obs_size) -> (batch_size, N, obs_size)
        repeated_obs_t = reshaped_obs_t.expand(
            x.shape[0], self._n_action_samples, *x.shape[1:]
        )
        # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
        flat_obs_t = repeated_obs_t.reshape(-1, *x.shape[1:])

        # (batch_size * N, 1)
        flat_values = self._q_func(flat_obs_t, flat_actions)
        # (batch_size * N, 1) -> (batch_size, N)
        reshaped_values = flat_values.view(x.shape[0], -1)

        # re-sampling
        probs = F.softmax(reshaped_values, dim=1)
        indices = torch.multinomial(probs, 1, replacement=True)

        return actions[torch.arange(x.shape[0]), indices.view(-1)]

    def sync_critic_target(self) -> None:
        assert self._targ_q_func is not None
        assert self._q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    def sync_actor_target(self) -> None:
        assert self._targ_policy is not None
        assert self._policy is not None
        hard_sync(self._targ_policy, self._policy)

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