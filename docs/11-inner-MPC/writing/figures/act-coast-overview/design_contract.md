# Figure 1: concept first, construction second

The figure should first explain what act–coast control does, then show how the controller constructs one event. Panel a carries the concept. Panel b uses a left-to-right sequence of timing samples, parallel conditional MPC solves, outer selection and execution, with neural information entering from above. All timing bars and input traces are schematic; they make no experimental claim.

## Retained text and its purpose

| Text retained in the figure | Necessary purpose |
|---|---|
| `a Act–coast control` | Establish the concept before introducing algorithms. |
| `Act`, `Coast` | Name the two colored phases. |
| `Input u(t)`, `Mode m(t)` | Distinguish commanded input from actuator enablement. |
| `u = 0` inside the act interval | Show that a zero command can occur while enabled. |
| `m = 1`, `m = 0` | Identify enabled and disabled operation. |
| `0`, `τ`, `h` | Locate the decision, shut-off and next decision on one elapsed-time axis. |
| `Plan`, `Shut off`, `Replan` | Explain the function of each timing marker. |
| `b Timing search and parallel MPC` | Introduce the construction after the concept. |
| `State x` | Identify the common observed state used for all conditional problems. |
| `Sample timings` | Explain the different act/coast duration pairs illustrated by the miniature bars. |
| `Parallel inner MPC` | Identify one independently optimized input sequence per timing sample. |
| `Actor: candidates` | State the actor's only displayed role without adding a training branch. |
| `Critic: terminal cost F` | Identify the common learned continuation information supplied to both optimization levels. |
| `Outer selection` and `min S` | Explain comparison of the optimized conditional event scores. |
| `τ*, h*, u*` | Identify the selected shut-off time, replanning time and corresponding optimized input sequence. |
| `Execute` | Connect the selected event to the act–coast sequence explained above. |

The caption defines the physical state, times, input, mode, critic estimate, conditional score and star notation. Teacher demonstrations and actor–critic training remain in Methods. Remove full-sentence box descriptions, the offline training pipeline, the separate plant box, return-loop narration, repeated continuation annotations and duration-grid algebra from the artwork. Place the formulation status in the caption. The figure now has 24 text boxes and 52 whitespace-separated text tokens, compared with 35 and 142 before this revision; the count includes symbols and panel letters.

## Composition and meaning

Use Morandi blue for enabled operation and muted orange for coast. Their pale fills retain the same meaning; black and neutral gray are reserved for text, arrows and outlines. No accent hue is required. Use Arial only, with 9-point headings and 8-point supporting labels, and black text throughout.

All workflow arrows use the same black stroke and arrowhead, directed rightward or downward. Three illustrated timing samples feed three matching MPC problems in parallel, each with the same initial state and its own optimized enabled input prefix. A network icon identifies each learned component. The miniature event and input illustrations replace prose rather than decorate it.

The shared critic estimate F is evaluated at each conditional prediction's replanning state. It is the inner terminal cost and is already included in the optimized score S used for timing comparison. Its two arrows indicate these two roles, not two additive penalties or a separate timing-policy network. The shown parallel evaluation is part of the proposed full controller, not a claim that the act–coast executor has been evaluated.

## Milestones, structure and success checks

1. **Scientific and text contract:** every label has the purpose listed above; concept comes first; both timing variables are sampled; each timing pair has a separate conditional input solve; critic dependence is shared without double counting.
2. **Editable rendering:** preserve the existing two scripts and three Python functions (`build_scene`, `render_vector`, `main`), adding only an ellipse primitive to the shared geometry/export vocabulary. No classes, wrappers, controller changes, runtime switches, tunable parameters or fallback behavior are added. Geometry, color, font and output paths remain fixed production values.
3. **Visual validation:** inspect both panels, arrow directions, text clearance, miniature timing/input correspondence, exactly two font sizes and black text; confirm only two semantic hues and neutral strokes. Audit vector fonts and separate native objects.
4. **Delivery:** update the editable Canva diagram and local vector exports, revise the bilingual caption, and compile the existing LaTeX source. Keep the human guide separate from geometry and validation records.
