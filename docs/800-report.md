# Burst-Coast MPC Handoff Report

## Project Purpose

This project studies burst-coast model predictive control for weakly actuated nonlinear systems, using the inverted pendulum as the current benchmark. The central problem is that a conventional short-horizon controller can be too myopic for swing-up, while a long-horizon controller is computationally expensive. The project separates fast physical actuation from slower strategic timing decisions: MPC remains responsible for torque planning, while a learning layer decides when and how long the controller should actively plan.

## Benchmark

The pendulum is intentionally underpowered, with torque limit below the gravitational torque scale. It cannot be driven directly to upright from all states; it must exploit natural dynamics, phase timing, and energy pumping. This makes it a useful minimum benchmark for the broader burst-coast idea: compute a short active burst, allow passive or zero-input coast, then replan only at meaningful decision epochs. The intended scientific claim is that learning timing over a burst-coast MPC structure can improve closed-loop behavior, actuation economy, and computation use for certain tasks.

## Problem Formulation

The plant state is

$$
x = [\theta,\omega]^T,
$$

with upright target $\theta=0,\omega=0$. The continuous dynamics are

$$
\dot{\theta}=\omega,\qquad I\dot{\omega}=mgl\sin\theta-b\omega+u,
$$

with $|u|\le u_{\max}$ and $u_{\max}<mgl$. MPC prediction uses a fixed-step RK4 discrete map. Mechanical energy is

$$
E(x)=\frac{1}{2}I\omega^2+mgl(1+\cos\theta),
$$

with target energy $E^\star=2mgl$ and normalized error $e_E=(E-E^\star)/E^\star$.

## MPC

The active MPC formulation uses a simplified energy-based cost. The current inner MPC objective is frozen around the working terms

$$
J_{\mathrm{MPC}}=\sum_i q_E e_E(x_i)^2+q_{\Delta u}\bar{\Delta u}_i^2,
$$


## Reinforcement Learning 

The current learning target is a Q-cost critic over a two-dimensional discrete action grid:

$$
a_k=(\bar B_k,\bar H_k),\qquad Q_\psi(s_k)\in\mathbb R^{N_B\times N_H}.
$$

Here $\bar H$ selects normalized prediction horizon up to the natural-period horizon, and $\bar B$ selects normalized burst length up to half of the selected horizon. The reduced observation is

$$
s_k=[\sin\theta_k,\cos\theta_k,\omega_k,|u_k/u_{\max}|]^T.
$$

MPC still solves torque. RL only chooses the burst and horizon pair. Deployment selects

$$
a_k^\star=\arg\min_a Q_\psi(s_k,a),
$$

while offline training uses weighted stochastic exploration over the same action matrix. The Q target represents expected closed-loop cost-to-go, including physical time, accumulated actuation, computation time, and terminal failure penalty.

## Progress
The current active step is Monte Carlo data generation for offline RL. The planned path runs the standalone MPC simulation under sampled $(\bar B,\bar H)$ policies, logs one RL transition per replanning epoch, and computes full Monte Carlo returns backward at episode end:

$$
G_k=\sum_{i=k}^{T}\gamma^{i-k}c_i.
$$

This creates the initial dataset $\mathcal D_{\mathrm{MC}}=\{(s_k,a_k,G_k)\}$ for critic pretraining. After that, the intended sequence is Q-network design, offline critic training, compute-time model fitting from solve logs, and then online simulated training while the MPC simulation still preserved as an independent controller.
