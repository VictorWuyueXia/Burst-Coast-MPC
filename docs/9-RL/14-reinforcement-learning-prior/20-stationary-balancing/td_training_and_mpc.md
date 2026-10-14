# TD training, Q-values and an MPC terminal cost

**Yes: our stationary TD3 learns two Q-functions, saved with the actor. They are candidate MPC terminal values, but their accuracy for MPC has not yet been validated.**

**How training works.** The actor maps five physical-state features to normalized torque. Each critic also receives an action and estimates Q: the discounted future reward from taking that action, then following the learned policy. Q is not necessarily the optimal value.

Let x be measured physical state, a applied torque divided by its cap, r measured reward integrated over 100 ms, and x next the simulated next state. With configured discount gamma=0.995 per action, temporal-difference training uses:

$$
y=r+\gamma\min_{i\in\{1,2\}}Q'_i(x_{\mathrm{next}},a').
$$

i indexes the critics; primes denote slowly updated target networks. The next action a prime comes from the target actor, with small clipped noise and motor limits. Critics minimize squared error between Q(x,a) and y. Every second critic update, the actor learns to increase predicted Q, with an additional teacher-imitation penalty. Demonstrations initialize training and remain mixed into replay. Success does not terminate learning; sampling timeouts still bootstrap from the real next state.

**Using it in MPC.** Rewards are negative costs: higher Q is better. Let pi denote the learned actor. Define terminal cost F and the cost-minimizing MPC objective J:

$$
F(x)=-\min_i Q_i(x,\pi(x)),\qquad
J=\sum_{k=0}^{H-1}\gamma^k\ell(x_k,a_k)+\gamma^H F(x_H).
$$

H is the chosen number of 100 ms prediction steps; k indexes predicted states x k and candidate actions a k. Stage cost ell equals minus our training reward. F estimates the remaining cost after the predicted segment. The critic minimum is not a guaranteed bound.

**For burst/coast MPC:** match features, torque cap, costs and discount. Predict idle intervals explicitly, then evaluate Q where feedback resumes: this Q assumes subsequent 100 ms control, not arbitrary sleep. First validate Q against independent discounted policy rollouts over MPC-reachable states, especially near arm limits. Successful actor control alone does not prove critic accuracy or MPC stability.

[Implementation](../../src/rotary_pendulum/RL/jax_td3.py) · [Detailed protocol](machine-scannables/protocol.md) · [Saved checkpoints](../../artifacts/rotary_pendulum/stationary-balancing-20261013/machine-scannables/selected_models.json).
