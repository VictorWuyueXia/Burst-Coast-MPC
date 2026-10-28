# Standing writing guidance

Recorded from the author's instructions on 2026-10-26. Apply these preferences to subsequent writing tasks. These instructions refine the exposition of the existing scientific content; they do not authorize changing the formulation or its evidence. Use the control-academic-writing skill with these author instructions taking precedence.

- Use simple grammar and one continuous line of reasoning. Introduce each concept because the preceding argument creates a need for it.
- Assume that the reader barely knows the task or the project. Explain the problem and the reason for each construction before presenting its formulation.
- Explain why a choice is made, rather than merely listing what was implemented.
- Keep theoretical and methodological prose symbolic. Omit configured numerical values and unit inventories from that prose; place experiment-specific settings and measured results in **Experiments and Results** or the reproducibility record. Retain mathematical constants required by the equations.
- Define every equation, symbol, and intermediate relation when introduced. Immediately explain the physical or control-theoretic intuition and how the relation advances the argument. Walk the reader through necessary derivation steps.
- Use the smallest amount of mathematics that states the method accurately. Prefer a clear compact relation to a dense expansion. Explain `normalize()` once when exact scaling is peripheral; retain an explicit scaling when it affects the mathematical claim. Avoid families of similar symbols distinguished only by multiple subscripts.
- Use transferable control-theory terminology. Avoid “capture” and “window”; choose the precise intended term, such as reaching upright, stabilization, holding, prediction horizon, or evaluation interval.
- Keep experimental observations and performance claims in **Experiments and Results**. Methods explain the construction and its mathematical properties.
- Give each paragraph one topic and one purpose. Usually use three to five sentences: open with the intuition or topic, develop the necessary reasoning, and end with its implication or the reason for the next paragraph.

For each planned paragraph, identify its opening idea, sentence sequence, equation placement, immediate interpretation, and connection to the next paragraph. Check that removing the paragraph would remove a necessary step. Keep a detailed source/evidence record separate from the reader-facing argument.

The theoretical exposition must preserve the distinction between a design objective, an analytical conclusion, and an observed result. Concision does not permit silently replacing the implemented controller with a simpler but different one.
