# Figure 4: linked keyframes and complete recorded control

Figure 4 is a wide double-column page-top figure. Its purpose is to show one recorded transition from swing-up to upright holding, connect the motion to the applied controller behavior, and reserve the matched act–coast comparison without inventing data. Complete scientific plots, labels and keyframe links are generated in Python; Canva composes the intact bases and leaves the pending comparison editable.

## Topic-by-topic visual contract

| Question | Why include it? | Clearest presentation | Arrangement with other topics |
|---|---|---|---|
| What physical motion do the traces describe? | The benchmark coordinates are otherwise abstract. | Four fixed-camera recorded configurations: initial state, largest arm excursion before first upright entry, first entry, final state. | Across the top, located by the same piecewise-linear time mapping as the main traces. Their letters and exact times connect through thin neutral vertical guides. |
| How does the full state evolve? | Upright orientation alone omits arm motion and residual velocities. | Separate pendulum-angle and arm-angle rows, plus paired angular-rate traces, with descriptive normalized axes. | All rows share the same adjacent 0–8 and 8–40 second axes. Keep the centered continuous pendulum-angle branch. |
| How do energy and input authority change? | These diagnostics connect the record to Figure 3's mechanism. | Total and relative-swing energy, followed by their direct torque sensitivities. | Two aligned rows use blue for the first series and orange for the second; legends name each pair. |
| What does the controller execute? | Small torque is not evidence of coast. | Exact zero-order-held torque, an uninterrupted enabled rail, and the fixed replanning-interval annotation. | Immediately below the state and energy rows on the identical time mapping. Keyframe guides identify state landmarks, not controller events. |
| How small is the holding error? | The full-motion scale cannot resolve the tolerance. | A separate late-time upright-error detail with the assessment band and holding-completion marker. | Below the main record, with its own clearly labeled linear 8–40 second axis. Stop the main keyframe guides before this different time mapping. |
| What does Omega mean? | A scaling symbol must not require inference. | Define Omega as the coupled downward small-motion natural angular frequency. Verify its formula against the archived model. | A concise Python header note defines it; Methods and the caption state the mathematical definition and how it scales the rates. |
| Where will act–coast evidence go? | The figure must preserve the intended comparison while distinguishing absent data. | A labeled blank full-width comparison region. | Below the continuous record, with an editable pending-record note. No invented curves, zero traces, mode transitions or savings. |

## Dimensions and style

Use a 504 × 460 point paper canvas and a LaTeX figure* [t] float at textwidth. The main time views span x=80–350 for 0–8 seconds and x=373–478 for 8–40 seconds. The physical configurations use that exact map, including the actual final-time position. All 40,001 states and 400 applied held commands remain present. The holding detail has its own linear late-time scale.

Use Arial only, 8-point labels and 9-point headings, black text, Morandi blue #7B94A6 for every first/sole data series and orange #C49D7D for a second. Use thin paired traces with a secondary dash pattern where useful; keep reference lines and geometry neutral so color identifies plotted series rather than a global component identity. Do not add explanatory decorations that duplicate axes, legends or guide lines.

## Structure budget and milestones

Retain the existing load_session and derive_annotations decoder functions. Keep three substantive renderer functions: prepare_record verifies sources and diagnostics; draw_figure aligns geometry, plots and reference information; main exports paper vectors, two complete plot bases and the production audit. Retain one compact JavaScript exporter with two intact images and editable comparison text. No new classes, wrappers, CLI/config parameters, controller logic, simulation or training.

1. Verify the record and frequency formula. Success: canonical hashes, sample/command alignment, archived holding assessment and the coupled natural frequency agree.
2. Verify the drawing geometry and line semantics. Success: each keyframe center and guide shares its exact plotted timestamp, all scalar curves are blue, paired curves use blue/orange and thin strokes, and no record is sampled or simplified.
3. Inspect full-size paper and native previews. Success: descriptive labels fit, linked poses do not collide, legends clear data, Omega is defined, and the comparison remains visibly empty.
4. Import the editable composition into Canva, archive the superseded design, update the caption and compile the saved manuscript. Success: intact complete bases, correct folder organization, figure* [t] at textwidth, and built-in compilation succeeds. Compilation and full-manuscript rendered float placement are separate checks.
