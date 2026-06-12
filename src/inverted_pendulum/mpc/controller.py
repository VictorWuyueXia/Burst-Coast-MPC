"""Public MPC controller selected by configuration."""

from __future__ import annotations

import logging

from inverted_pendulum.mpc.ip_dynamics_natural_period.controller import NaturalPeriodMPCController
from inverted_pendulum.mpc.types import SelectedPlan
from inverted_pendulum.utils.config_schema import EnvironmentConfig, MPCConfig
from inverted_pendulum.utils.messages import ActionCommand, StateObs
from inverted_pendulum.utils.monte_carlo import MonteCarloAction


class CasadiMPCController:
    """Dispatch the configured CasADi MPC formulation through one stable interface."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        mpc: MPCConfig,
        *,
        logger: logging.Logger,
    ) -> None:
        # The active configuration always dispatches to the natural-period formulation.
        self.controller = NaturalPeriodMPCController(environment, mpc, logger=logger)

    def select_action(self, observation: StateObs, *, force_replan: bool) -> ActionCommand:
        """Return one action from the configured concrete MPC controller."""

        return self.controller.select_action(observation, force_replan=force_replan)

    def start_monte_carlo_plan(
        self,
        observation: StateObs,
        monte_carlo_action: MonteCarloAction,
    ) -> SelectedPlan:
        """Start one sampled Monte Carlo plan on the concrete MPC controller."""

        return self.controller.start_monte_carlo_plan(observation, monte_carlo_action)
