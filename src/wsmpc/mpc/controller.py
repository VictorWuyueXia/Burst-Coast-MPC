"""Public MPC controller selected by configuration."""

from __future__ import annotations

import logging

from wsmpc.mpc.ip_dynamics_natural_period import CONFIG_NAME as NATURAL_PERIOD_NAME
from wsmpc.mpc.ip_dynamics_natural_period.controller import NaturalPeriodMPCController
from wsmpc.mpc.ip_energy_event_triggered import CONFIG_NAME as ENERGY_EVENT_NAME
from wsmpc.mpc.ip_energy_event_triggered.controller import EnergyEventMPCController
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
        if mpc.controller == ENERGY_EVENT_NAME:
            self.controller = EnergyEventMPCController(environment, mpc, runtime, logger=logger)
        elif mpc.controller == NATURAL_PERIOD_NAME:
            self.controller = NaturalPeriodMPCController(environment, mpc, runtime, logger=logger)
        else:
            msg = f"Unknown MPC controller: {mpc.controller}"
            raise ValueError(msg)

    def select_action(self, observation: StateObs) -> ActionCommand:
        """Return one action from the configured concrete MPC controller."""

        return self.controller.select_action(observation)
