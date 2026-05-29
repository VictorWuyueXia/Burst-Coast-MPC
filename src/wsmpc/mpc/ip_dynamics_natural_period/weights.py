"""Fixed cost constants for the natural-period dynamics MPC."""

# Phase normalization uses a fixed radius floor to keep the proxy differentiable.
EPSILON_PHI = 1.0e-6

# Distribution scales shape smooth activation profiles around the target energy shell.
signal_energy = 0.25

# Cost weights scale scalar objective terms while matrix weights stay as Q.
w_energy = 1.0
w_phase = 0.0
w_local = 0.0
w_terminal = 0.0
w_saturation = 0.0
w_delta_u = 1.0e-3

# Diagonal quadratic forms keep phase and local state metrics explicit.
Q_PHASE_DIAG = (1.0, 1.0)
Q_LOCAL_DIAG = (1.0, 1.0)
