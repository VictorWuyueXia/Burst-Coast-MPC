"""CasADi dynamics expressions matching the simulator's RK4 pendulum model."""

from __future__ import annotations

from typing import Any

import casadi as ca

from wsmpc.utils.config_schema import PendulumConfig


def pendulum_derivatives_symbolic(x: Any, u_nm: Any, pendulum: PendulumConfig) -> Any:
    """Build continuous-time pendulum derivatives with theta zero at upright."""

    theta_rad = x[0]
    omega_rad_s = x[1]
    inertia = pendulum.mass_kg * pendulum.length_m**2
    omega_dot = (
        pendulum.mass_kg * pendulum.gravity_m_s2 * pendulum.length_m * ca.sin(theta_rad)
        - pendulum.damping_nms * omega_rad_s
        + u_nm
    ) / inertia
    return ca.vertcat(omega_rad_s, omega_dot)


def rk4_step_symbolic(x: Any, u_nm: Any, timestep_s: float, pendulum: PendulumConfig) -> Any:
    """Build one fixed-step RK4 map from continuous pendulum dynamics."""

    # Each intermediate derivative shares the same constant control over the sample.
    k1 = pendulum_derivatives_symbolic(x, u_nm, pendulum)
    k2 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k1, u_nm, pendulum)
    k3 = pendulum_derivatives_symbolic(x + 0.5 * timestep_s * k2, u_nm, pendulum)
    k4 = pendulum_derivatives_symbolic(x + timestep_s * k3, u_nm, pendulum)
    return x + (timestep_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_rk4_function(timestep_s: float, pendulum: PendulumConfig) -> ca.Function:
    """Create a small CasADi function for parity tests and diagnostics."""

    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u")
    return ca.Function("pendulum_rk4", [x, u], [rk4_step_symbolic(x, u, timestep_s, pendulum)])
