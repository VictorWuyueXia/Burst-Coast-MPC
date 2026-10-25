# Interpretation of the MPC+RL proof review

Reviewed on 2026-10-24 against commit `cdce8f93f98d6c978e6a723d4aea11f4524c59ad` and the two local `11.1-*.md` discussions. The commit is the current checkout; the discussions are untracked additions to it.

**Judgment: the conditional capture argument is mathematically sound, but it does not yet prove convergence of this controller to an upright holding region.** Its main assumptions are precisely the difficult controller properties that remain unestablished: uniform progress and continued accepted execution. The separate holding inequality is also unverified.

The stage-cost threshold and exact-ODE lower-bound lemma are valid supporting results. They cannot replace the missing connection between the frozen RL value, CEM selection, executed dynamics, and closed-loop progress. The proof is a useful certificate template, not a completed controller theorem or evidence that RL is necessary.

Read [the assessment](human-readables/proof_assessment.md) for the mathematical judgment and [the obligations](machine-scannables/proof_obligations.md) for exact missing inputs, acquisition steps, and success criteria.

Recommended order: recover the frozen checkpoint and records; define a meaningful holding domain under the actual MPC; test capture and progress into that domain; establish continued accepted execution and rigorous regional coverage; compose the two guarantees with compatible controller histories. Counterexamples must be reported before changing the controller or the certificate.

The checkpoint and numerical experiment archives are absent from this checkout. No controller experiment was rerun. Source inspection and independent arithmetic confirm the stated stage-cost threshold and mechanical constants; reported 32-session outcomes remain documentation-derived evidence.
