"""Public MPC controller selected by configuration."""

from __future__ import annotations

import logging

from wsmpc.mpc.ip_dynamics_natural_period.controller import NaturalPeriodMPCController
from wsmpc.utils.config_schema import EnvironmentConfig, MPCConfig, RuntimeConfig
from wsmpc.utils.messages import ActionCommand, StateObs


class CasadiMPCController:
    """Dispatch the configured CasADi MPC formulation through one stable interface."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        mpc: MPCConfig,
        runtime: RuntimeConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        self.controller = NaturalPeriodMPCController(environment, mpc, runtime, logger=logger)

    def select_action(self, observation: StateObs, *, force_replan: bool) -> ActionCommand:
        """Return one action from the configured concrete MPC controller."""

        return self.controller.select_action(observation, force_replan=force_replan)
