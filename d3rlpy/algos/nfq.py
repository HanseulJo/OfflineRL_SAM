from typing import Any, Dict, Optional, Sequence, List

from ..argument_utility import (
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.dqn_impl import DQNImpl
from ..iterators import TransitionIterator


class NFQ(AlgoBase):
    r"""Neural Fitted Q Iteration algorithm.

    This NFQ implementation in d3rlpy is practically same as DQN, but excluding
    the target network mechanism.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \max_a Q_{\theta'}(s_{t+1}, a) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Riedmiller., Neural Fitted Q Iteration - first experiences with a
          data efficient neural reinforcement learning method.
          <https://link.springer.com/chapter/10.1007/11564096_32>`_

    Args:
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.dqn_impl.DQNImpl): algorithm implementation.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _use_gpu: Optional[Device]
    _impl: Optional[DQNImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[DQNImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        ######## For SAM ##########
        if isinstance(loss, tuple):
            loss, sharpness = loss
            metrics = {"loss": float(loss), "sharpness": float(sharpness)}
        else: metrics = {"loss": float(loss)}
        ###########################
        self._impl.update_target()
        return metrics # For SAM  # originally {"loss": float(loss)}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE
    
    def _hessian_max_abs_eigs(self,
        iterator: TransitionIterator,
        top_n: int,
        max_iter: int,
        tolerance: Optional[float],
        show_progress: Optional[bool],
    ) -> Dict[str, List[float]]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert top_n > 0
        hessian_top_n_eigenvalues = self._impl.hessian_eig_imitator(iterator, top_n, max_iter, tolerance, show_progress)
        return {'hessian_top_n_eigenvalues': hessian_top_n_eigenvalues}

    def _hessian_spectra(self,
        iterator: TransitionIterator,
        n_run: int,
        max_iter: int,
        show_progress: Optional[bool]
    ) -> Dict[str, List[List[float]]]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        hessian_eigenvalues, weights = self._impl.hessian_spectra_imitator(iterator, n_run, max_iter, show_progress)
        return {'hessian_spectra': (hessian_eigenvalues, weights)}
