"""Direct-torque PPO training and artifact support for the rotary pendulum."""

from rotary_pendulum.RL.environment import BatchedRotaryPPOEnvironment
from rotary_pendulum.RL.model import RotaryActorCritic

__all__ = ["BatchedRotaryPPOEnvironment", "RotaryActorCritic"]
