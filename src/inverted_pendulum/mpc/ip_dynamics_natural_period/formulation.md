# IP Dynamics Natural-Period MPC Formulation

This controller solves one single-shooting nonlinear program for each configured split ratio and executes the minimum-cost burst-coast plan.

## Horizon and Decision Variables

The horizon is one full small-angle natural period,

```math
\omega_n = \sqrt{\frac{mgl}{ml^2}}, \qquad
T_n = \frac{2\pi}{\omega_n}, \qquad
N = \operatorname{round}\left(\frac{T_n}{\Delta t}\right).
```

For a split ratio `lambda`, the actuated burst and passive coast lengths are

```math
N_b = \operatorname{round}(\lambda N), \qquad N_c = N - N_b.
```

The NLP decision vector is the burst torque sequence,

```math
v = [v_0,\ldots,v_{N_b-1}]^\top.
```

The applied prediction input is

```math
u_k =
\begin{cases}
v_k, & 0 \le k < N_b,\\
0, & N_b \le k < N.
\end{cases}
```

## Dynamics

The state is `x = [theta, omega]^T`, with `theta = 0` at the upright equilibrium. The continuous dynamics used inside the RK4 step are

```math
\dot{\theta} = \omega,
```

```math
\dot{\omega} =
\frac{mgl\sin\theta - d\omega + u}{ml^2}.
```

The prediction model is the fixed-step RK4 map

```math
x_{k+1} = f_d(x_k, u_k; \Delta t),
```

with initial condition `x_0` equal to the observed state.

## Energy-Phase Value

The upright target energy and normalized energy error are

```math
E^\star = 2mgl,
```

```math
e_E(x) =
\frac{0.5ml^2\omega^2 + mgl(1 + \cos\theta) - E^\star}{E^\star}.
```

The energy-shell activation is

```math
g_E(x) = \exp\left(-\frac{e_E(x)^2}{\sigma_e^2}\right).
```

The phase proxy is

```math
a_\theta = \cos(0.5\theta), \qquad
b_\phi = \frac{\omega}{\omega_n},
```

```math
r_\phi = \sqrt{a_\theta^2 + b_\phi^2 + \epsilon_\phi^2},
```

```math
z_\phi(x) =
\begin{bmatrix}
a_\theta/r_\phi - 1\\
b_\phi/r_\phi
\end{bmatrix}.
```

The local upright error is

```math
z_l(x) =
\begin{bmatrix}
\operatorname{atan2}(\sin\theta,\cos\theta)\\
\omega/\omega_n
\end{bmatrix}.
```

The state value is

```math
V(x) =
w_e e_E(x)^2
+ w_\phi g_E(x) z_\phi(x)^\top Q_\phi z_\phi(x)
+ w_l g_E(x) z_l(x)^\top Q_l z_l(x).
```

The implemented diagonal matrices are

```math
Q_\phi = \operatorname{diag}(1,1), \qquad
Q_l = \operatorname{diag}(1,1).
```

## Objective

For the burst segment, the stage cost adds saturation attraction and input smoothness:

```math
\ell_b(x_k,u_k,u_{k-1}) =
V(x_k)
+ w_u\left(1 - (u_k/u_{max})^2\right)^2
+ w_{\Delta u}\left((u_k-u_{k-1})/u_{max}\right)^2.
```

Here `u_{-1}` is the last torque executed before solving the current plan. For the coast segment,

```math
\ell_c(x_k) = V(x_k).
```

The optimized objective is

```math
J(v) =
\sum_{k=0}^{N_b-1} \ell_b(x_k,v_k,u_{k-1})
+ \sum_{k=N_b}^{N-1} \ell_c(x_k)
+ w_T V(x_N).
```

## Constraints

The decision variables are torque-bounded:

```math
-u_{max} \le v_k \le u_{max}, \qquad k=0,\ldots,N_b-1.
```

The coast inputs are fixed to zero by construction. The state trajectory is constrained only through the RK4 single-shooting rollout from the initial state.

## Solver

Each split-ratio candidate is built as a CasADi `MX` nonlinear program and solved with IPOPT. The solver uses silent output, `ipopt.max_iter = 100`, and `ipopt.tol = 1.0e-6`. Candidate NLPs are evaluated independently, and the controller selects the finite successful solution with minimum objective value.
