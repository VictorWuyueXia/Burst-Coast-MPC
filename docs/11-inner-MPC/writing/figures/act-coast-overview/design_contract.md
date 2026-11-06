# Figure 1: repeated act–coast operation and controller construction

The visual claim is that a finite set of timing-conditioned input optimizations produces one act–coast episode, and the construction repeats at later decisions. This is a schematic-led composite, not a measured controller record. Apply [the example-derived style rules](../figure_style_rules.md).

## Panels and text purposes

| Element | Scientific purpose | Necessary labels |
|---|---|---|
| Three episodes | Show densely sampled held commands during actuation and zero input during coast. Remove the separate Mode element entirely. | Act, Coast, input u(t), time t |
| Start markers and brackets | Distinguish activation count from accumulated enabled duration; show repeated decisions and each actuation interval. | t₁, t₂, t₃; τ₁, τ₂, τ₃; 3 activations; total actuation time |
| Timing samples | Show different act/coast duration pairs as small repeated timing diagrams. | Sample timings |
| Finite selection placeholder | Reserve an explicit place for a rule choosing a finite number of timing-conditioned inner problems. No sampling or ranking rule is invented. | Select subset; K pairs; Rule pending |
| Card stack | Show independent inner MPC+RL problems evaluated in one parallel batch, each associated with its own timing pair and optimized input sequence. | Solve in parallel; Inner MPC+RL; (τᵢ,hᵢ); min J+F |
| Actor and critic | Actor supplies input candidates. Critic F enters the terminal objective and therefore also the optimized scores compared by outer MPC. | Actor; Critic F |
| Outer decision and execution | Outer MPC chooses shut-off, replanning time and their compatible input sequence; execution proceeds to the next episode. | Outer MPC; min S; τ*, h*, u*; Execute |

The caption defines all symbols. K is a finite positive candidate count, with its selection rule unspecified. Three visible episodes and three stacked cards are illustrative, not an experimentally established count or batch-speed result. Blue denotes enabled actuation, orange coast; a zero command inside blue remains an enabled command without a separate mode trace.

## Rendering and structure budget

Keep Arial 8/9 points, black text, two Morandi hues and neutral strokes. Use 504 × 342 points at double-column width. Main arrows share one style and move rightward or downward. Stack offsets expose multiplicity; the front card contains a small optimization and input illustration. Group the dense actuation lines as one editable Canva element while preserving constituent pieces.

Retain build_scene, render_vector and main in the existing Python source, and the existing native exporter. Add only scene group metadata and native grouping to that exporter; no new classes, wrappers, experiment parameters, CLI switches or fallback behavior. Keep source geometry and audit data in artifacts and human guidance in docs.

## Milestones and success criteria

1. Contract: save example-derived rules and this label/purpose map before rendering. Success: every element carries a physical or algorithmic relation.
2. Artwork: remove Mode, draw dense commands for three episodes, bracket durations and count activations, add the finite selector and card stack. Success: labels and connections are readable at final size with no false algorithmic claims.
3. Editing: export native text/shape geometry and three native burst groups, then inspect Canva grouping and editability. Success: the action group moves/ungroups as one Canva object and current figure remains in act-coast with its predecessor archived.
4. Manuscript: update local vectors and bilingual caption; compile the existing main.tex. Success: compilation passes without claiming that the proposed timing controller has been evaluated.
