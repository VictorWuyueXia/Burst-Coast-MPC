# Figure 4: synchronized motion, control, and physical configurations

## Purpose and central statement

Show how one controller session produces swing-up and continued upright holding, and how the applied input and enabled mode relate to that motion. The future matched act–coast session should reveal where forced motion ends, how the system evolves without input, and when feedback resumes. This figure explains execution over time; Figure 1 explains the architecture, Figure 3 explains energy mechanisms, and Figure 5 evaluates resource use across scenarios.

The current figure can establish only the continuously enabled MPC+RL behavior in one fixed simulation session. The act–coast block remains empty until its matched record exists. Do not use the future comparison to imply savings, robustness, ideal rest, or constraint guarantees before those properties have evidence.

## Overall composition

Use a double-column page-top figure, approximately 7 inches wide. Start with an artwork height of about 5.3 inches, excluding the caption; retain Arial at 8 and 9 physical points rather than shrinking text to meet that estimate. The exact height is a production decision after a final-size layout check.

Arrange two session blocks vertically: **(a) Continuously enabled MPC+RL**, then **(b) Act–coast MPC+RL**. Each populated block contains a shallow strip of four physical keyframes, followed by vertically aligned signal rows. Keeping one controller within each block lets the reader connect its state, torque, and mode without separating overlapping controller curves.

Within the trace area, use two adjacent time views. Allocate roughly two-thirds of the horizontal plotting space to the expanded initial transient and one-third to the complete remaining record. Each view has an ordinary linear elapsed-time axis; the rates of horizontal scaling differ, and the boundary must be explicit. Include the boundary sample on both sides for continuity; omit no elapsed interval or data.

Label the two views **Expanded transient** and **Remaining record**, once across the full figure. Freeze their common division from the already selected continuous session before any act–coast result is inspected. Use the same division, full end time, horizontal allocations, and corresponding vertical ranges in both controller blocks. The second block is currently reserved whitespace with one brief pending-data label, not a speculative diagram.

The split solves a specific presentation problem: most of the current session follows initial swing-up. A uniform full-record time axis makes early motion and command changes difficult to distinguish; a transient-only crop would hide the evidence for continued holding. Adjacent explicitly scaled views retain both.

## Signal rows and their purposes

| Row | Include and presentation | Purpose |
|---|---|---|
| Upright error | Signed error relative to the ideal upright orientation, a thin zero reference, and the actual assessment tolerance. Show the large transient at left and a magnified tolerance-based ordinate at right. | Separate reaching upright from remaining within the prescribed error range; make late deviations visible rather than visually flattening them. |
| Arm travel | Signed unwrapped arm angle divided by its prescribed travel limit; identical scale in both time views; thin references at the two limits. | Show the excursion and braking required for swing-up, proximity to the arm boundary, and residual arm displacement during holding. |
| Angular speeds | Arm and pendulum speeds divided by one common model-derived frequency; black solid and gray dashed curves, with direct labels. | Distinguish upright orientation from ideal rest and show whether motion persists during holding or coast. |
| Applied input | The command actually applied, normalized by its torque cap; a black zero-order-hold staircase with references at zero and the input bounds. | Show command sign, amplitude, saturation, changes, and braking without presenting planned future commands as executed torque. |
| Mode and decisions | A narrow enabled/coast ribbon immediately below the torque row, with the feedback schedule on the same elapsed-time coordinates. | Distinguish zero commanded torque from disabled actuation, and distinguish replanning from off-to-on activation. |

Give the upright-error and torque rows slightly more height than the arm and speed rows. The mode ribbon is an annotation rail, not another large chart. Use one elapsed-time label below each controller block, with ticks shared vertically within each time view. Avoid repeated grid lines, repeated legends, and boxed subplot borders.

### Upright error and its two ordinates

Let alpha be the measured-from-downward pendulum angle. Define beta = wrap(alpha - pi), with wrap selecting the equivalent angle in [-pi,pi), and let epsilon > 0 be the angular tolerance already used to assess the recorded experiment. The left ordinate resolves the swing-up; the right ordinate resolves the assessment tolerance,

$$
\beta=\operatorname{wrap}(\alpha-\pi),\qquad
\text{left: }\frac{\beta}{\pi},\qquad
\text{right: }\frac{\beta}{\varepsilon}.
$$

The ideal goal is $\beta=0$ in both views; the right-view assessment band is exactly $[-1,1]$. Here $\varepsilon$ denotes the epsilon tolerance introduced in this paragraph.

This change of ordinate is deliberate, not automatic rescaling. Print the two ordinate definitions beside their respective axes and explain them in the caption. The time division is not the observed start of holding and must not be labeled as such. If a later matched act–coast record leaves the proposed right-view range, enlarge that range for both controllers and keep every excursion visible.

Insert line breaks where the wrapped error jumps across its branch cut; do not join opposite endpoints through the upright reference. The break affects only the visual connector, not the saved data or assessment samples. Draw the finite assessment band as a neutral light-gray region and the ideal reference as a thin line. The ideal-goal convention used in Figure 2 does not justify replacing measured finite-tolerance performance with exact-rest claims here.

### Arm travel and angular speeds

Use theta/theta_max for arm travel, where theta is the unwrapped arm angle and theta_max > 0 is the prescribed arm bound. Keep references at -1 and +1. Do not clip the trace to those references or describe a soft penalty as exact enforcement. An excursion beyond them must remain visible.

Use omega/Omega and nu/Omega for speeds, where omega = dtheta/dt, nu = dalpha/dt before angular wrapping, and Omega > 0 is the common downward small-motion frequency computed from the archived physical model. Preserve signed speeds and use the same scale for both signals and both controllers. The different line styles identify the two coordinates without adding another semantic hue. Show the speed row because a nearly vertical rod can coexist with continuing arm and pendulum motion.

### Torque, enabled mode, and feedback decisions

Let a = u/U be the applied torque u divided by its prescribed cap U > 0. Plot the held first command from each accepted decision, not every command in the optimized horizon. A sample at a command boundary uses the right-continuous input convention; document the interval endpoint convention in the source table. Never connect separate command values by a sloping line.

Let m(t) equal one while enabled and zero during coast. Use muted blue for m = 1 and muted orange for m = 0, matching the act/coast colors in Figure 1. The continuously enabled controller has one blue interval throughout its record, including times when a = 0. Its periodic feedback schedule is known from the archived runner; represent it with a compact cadence rail labeled “Every Delta,” using Delta as the fixed command interval, rather than hundreds of individually annotated markers.

For the eventual act–coast block, place black square ticks at actual replanning times and short vertical marks at actual shut-off times. These are discrete controller events, not torque zero crossings. Explain one recorded cycle with the current decision t_j, shut-off t_j + tau_j, and next decision t_j + h_j; tau_j > 0 is the enabled duration and h_j > tau_j is the time until replanning. Annotate only that one cycle with symbols; all other events retain their compact marks. Keep the current blank block free of event marks, colored mode intervals, and fictitious cycle durations.

## Physical keyframes

Use four compact reconstructions above each populated session block. Inherit Figure 2's shaded geometric style: a simplified neutral servo base and shaft, a blue horizontal arm, the passive hinge, and a dark pendulum rod. Preserve the actual changing arm orientation and arm-normal pendulum plane. Use one camera, scale, physical origin, and link proportions for every keyframe and both controllers; do not rotate the apparatus separately to make each pose look clearer.

| Keyframe | Fixed selection rule | Information conveyed |
|---|---|---|
| A | Actual saved initial state. | Establish the common initial condition and the task to be accomplished. |
| B | Earliest sample attaining the largest absolute arm excursion before the continuous session first enters its upright tolerance band. | Connect visible arm travel and a nearby braking action with the operating constraint. Do not call it a shut-off or optimum-energy event. |
| C | First sampled entry of that continuous session into the upright tolerance band. | Show the physical configuration corresponding to the first upright entry; entry alone does not establish completion of the required hold. |
| D | Actual final saved state of the full assessment. | Connect late holding and remaining arm displacement with a physical configuration. |

Freeze A–D from the continuous session and use those same elapsed timestamps for the future act–coast poses. This compares the controllers at the same times instead of choosing a favorable event separately for each controller. Their own hold-completion and switching events remain annotated on their respective traces. Retain an unsuccessful or not-yet-upright act–coast pose if that is what the matched record contains.

Space the four miniature drawings evenly for readability, with the letter and actual elapsed timestamp below each. State that this strip is a labeled sequence, not a time-proportional axis. Place matching A–D marks on the trace timeline; use fine vertical guides only at B and C, where their connection to transient state and input is useful. Avoid long diagonal connector lines from the pose strip to crowded early-time samples.

Use the recorded state at each marked time and the benchmark's mechanism mapping. The drawings represent simulated configurations, not photographs or new hardware evidence. Do not add trajectory trails, cartoon arrows, repeated angle labels, or velocity arrows: the neighboring traces already supply those quantities.

## Holding annotation

Distinguish first tolerance entry, the start of a qualifying holding interval, and confirmation of its required duration. Let H > 0 be the experiment's prescribed holding duration. Define t_s as the earliest recorded start for which every assessed sample through t_s + H satisfies |beta| <= epsilon; define t_c = t_s + H as its completion time. The data processing must use the archived sampled-time convention exactly. For recorded sample times in the finite set $\mathcal S$, the start condition is

$$
|\beta(s)|\leq\varepsilon
\quad\text{for every }s\in\mathcal S\cap[t_s,t_s+H],
\qquad t_c=t_s+H.
$$

Require the complete duration $H$ to lie within the available record. This is verification at recorded times, not a continuous-time bound between samples.

Mark t_s with a small neutral tick, merging it with C when those times coincide, and mark t_c on the upright-error row with the brief label **Hold criterion met**. Do not draw a duration bracket across the change of time scale; its paper length would give a misleading duration cue. Assess arm admissibility separately; the orientation criterion must not silently acquire a zero-speed requirement. If the matched session has no qualifying hold, omit the completion tick and retain the failed trajectory rather than invent a completion time.

## Palette, typography, and retained text

Use Figure 1's Morandi blue #7B94A6 and orange #C49D7D, with white, black, and neutral gray. Use only Arial and the two established physical sizes, 8 and 9 points. Blue identifies the driven arm in poses and enabled mode in the rail; orange identifies coast. Keep signal traces neutral so their color does not incorrectly suggest that a signal disappears during coast. Keep all text black except a label explicitly tied to one of those concepts.

Retain only the two controller names, the two time-view labels, signal/normalization labels, essential ticks, the four keyframe letters and timestamps, compact speed labels, one cadence label, one holding annotation, and one example act–coast-cycle annotation when data exist. Each retained text group identifies a quantity, time, event, or distinction needed to interpret the record. Keep algorithm explanations, solver settings, aggregate performance numbers, training information, and neural-network diagrams out of the artwork.

## Present data and current evidence

Use the already fixed downward session 0 from the official continuously enabled confirmation. The raw record is available locally and includes all selected plans and executed states. A read-only check confirmed a complete record with its actual initial condition, a brief swing-up transient, and a substantially longer upright segment with nonzero residual speeds. These observations motivate the expanded transient and the speed row, not a change of session selection.

For now, populate only block (a). Reserve block (b) with its heading and one small sentence, **Matched act–coast record pending**. Leave its pose and trace interiors empty. Do not draw a hypothetical sparse input, copy the continuous trajectory, interpolate a future result, or infer enabled mode from torque magnitude.

Do not include solve-time curves: the archived timing is for a batched computation and is not a single-session feedback latency. Do not add activation totals, enabled-time comparisons, or baseline rankings; those need the matched scenario accounting planned for Figure 5.

## Caption and narrative placement

Place the figure in Experiments and Results, next to the current continuously enabled MPC+RL performance subsection. The caption should introduce the fixed session, identify the two time scales and the two error ordinates, define the assessment references and event marks, explain A–D and the shared-timestamp rule, and identify the blank act–coast block. State that the figure is one simulated session; aggregate evidence belongs to the separate scenario figures.

The reader should be able to answer four questions without searching elsewhere: Did the pendulum reach and hold near upright? What arm travel and residual motion accompanied that behavior? What torque and feedback activity produced it? Which portions of the eventual matched record are unforced? The current figure answers the first three and visibly reserves the fourth.

## Milestones and success checks

| Milestone | Work | Success check |
|---|---|---|
| Record contract | Verify archive identities, initial condition, sample times, applied-command decoding, and assessment rule. | All displayed observations trace to one complete fixed session; no future plan variable is mistaken for an executed command. |
| Layout specification | Freeze the common time division, row order, scales, keyframe rules, and style. | Every row aligns in elapsed time; transient and full remainder are readable; the second block is empty. |
| Numerical extraction | Derive error, normalized signals, keyframe states, and hold events from preserved records. | Match archived hold duration; preserve branch cuts and every out-of-bound observation; poses reproduce their exact samples. |
| Rendering and editable export | Draw the native vector figure and inspect at actual double-column width. | Arial 8/9 points, no overlap or clipped extrema, faithful held inputs, preserved curves/text in Canva, and correct physical PDF size. |
| Manuscript integration | Insert the bilingual caption and double-column top float in the existing manuscript. | Saved source compiles; later inspect the complete rendered page before claiming layout readiness. |

The subsequent author instruction authorizes production from these records and assembly in Canva. Controller simulations and synthetic act–coast records remain outside this figure task. Production uses the same signal and keyframe plan; artwork height expands to preserve readable 8/9-point text and the blank matched block.

## Planned implementation budget

Keep a minimal continuous workflow and no runtime CLI/configuration or fallback behavior. Plan one extraction module with two functions, load_session and derive_annotations; one drawing module with two functions, build_scene and render_vector; and one native editable exporter adapted from the established scene format. Add no classes, data classes, thin wrappers, controller components, or new experiment parameters. Fixed figure constants are paths, selected lane, time division, geometry camera, palette, font sizes, and layout dimensions. Function inputs are the verified session arrays, archived physical metadata, and derived annotations; return plain arrays and dictionaries. Preserve source data and QA under artifacts, separate from the human-facing plan and interpretation guide.
