# Figure 4 record, transforms, and validation contract

## Authority and frozen selection

The author subsequently authorized record-derived figure production and Canva assembly. New controller simulation and invented act–coast data remain outside scope. Current source authority is the official continuous MPC+RL confirmation, not the older portable teacher replay convention. In particular, do not reuse the older [0,2pi) pendulum-angle convention.

Repository-relative sources:

- Official transfer manifest: artifacts/rotary_pendulum/final-results/machine-scannables/file_manifest.json.
- Raw complete record: artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records/chunk-0000.npz through chunk-0039.npz.
- Archived design and source snapshots: design.json and sources/ in that same raw directory.
- Actual initial conditions: artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/session-records/starts.npz.
- Existing comparison-free metrics: artifacts/rotary_pendulum/final-results/rl-mpc/machine-scannables/episodes.csv and summary.json.

Select lane 0, already used as the downward representative; do not search for a more attractive result. The saved group index is 0. A read-only inspection found the actual initial state [0.06535831996416491, 0.06806342455862913, 0.13459179470736138, -0.01850166246763489], ordered as theta, alpha, omega, nu in radians and radians per second. It is near downward, not exact downward rest.

Freeze the time division at 8 seconds, with full endpoint 40 seconds. The first view is [0,8] and the second [8,40], each linear internally and occupying 270/105 points of plotting width, separated by a 23-point display gap. Duplicate only the 8-second endpoint across views, with no omitted duration. This division uses the existing continuous record only; do not optimize it against future act–coast outcomes.

## Verified array semantics

Each raw chunk has executed shape (10,32,100,4), plans shape (10,32,11), accepted shape (10,32), and solve_s shape (10,). Concatenation in filename order gives 400 decisions. For lane 0, flatten only decision and within-command sample dimensions, yielding 40,000 post-step states. Prepend the actual saved initial state at time zero; saved executed states occur at 0.001 through 40.000 seconds. The command interval is 0.1 seconds.

The first ten plan entries are normalized future inputs and the last is a terminal epigraph scalar. The applied normalized command is plans[k,0,0] in the concatenated decision array, held on [0.1k,0.1(k+1)). Use the archived plant torque_limit_nm to recover physical torque; do not silently substitute the current mutable runtime configuration or the teacher's separate cap. The final plotted endpoint uses the preceding held value at 40 seconds and does not create another decision.

The archived runner returns the refined accepted plan's first-command trajectory; confirm selected plans, accepted flags, and saved states refer to the same accepted choice before rendering. Mode is enabled throughout by this controller's execution definition. That fact is a source-derived schedule, not a recorded binary mode sensor, and must be identified as such in machine provenance. Actual act–coast mode and event logs are required for the future block.

solve_s contains elapsed time for a batch of 32 sessions, not one lane's solve time. Do not plot it as feedback latency for this chosen session, divide it by 32 and call the result latency, or convert the trace into a computation-savings comparison.

## Required transforms and quantities

Let alpha be pendulum angle from downward, theta the unwrapped arm angle, and omega, nu their signed speeds. Use

$$
\beta=\operatorname{atan2}(-\sin\alpha,-\cos\alpha).
$$

Map the positive-pi branch endpoint to the chosen [-pi,pi) convention if present. Use a continuous unwrap(alpha) branch in the main angle trace and verify its sine and cosine match the saved angle. Retain wrapped beta for assessment and the late detail; keep every sample. Do not unwrap theta, differentiate wrapped alpha, interpolate a different state, or force a small-angle approximation during swing-up.

The current archived tolerance is 5 degrees, arm design bound 135 degrees, required contiguous holding duration 10 seconds, and assessment duration 40 seconds. Convert these exact recorded criteria to radians for computation. Plot unwrap(alpha)/pi in both adjacent time views, anchored to the saved initial angle, and beta/epsilon in a separate full-width late-holding detail, where epsilon is the recorded angular tolerance. Plot theta/theta_max, omega/Omega, nu/Omega, and a=u/U as defined in the human plan. Derive Omega from the archived model's downward coupled small-motion frequency, retaining its formula and input coefficients in numerical provenance.

A read-only lane-0 check found first tolerance entry and start of uninterrupted remaining holding at 4.675 seconds; the saved total and maximum held durations are both 35.326 seconds. Peak absolute arm angle is approximately 2.3323342752645524 radians. Final-ten-second arm and pendulum speed RMS are approximately [0.29219041703000564,0.1279765657059509] radians per second. These values justify transient expansion and a speed row; they are not new multi-scenario results and do not establish ideal-rest convergence.

The earliest qualifying hold start t_s and completion t_c=t_s+H must be computed from the archived sample and endpoint convention. Preserve the distinction between sample count times dt and differences of inclusive endpoint timestamps; verify agreement with the existing report rather than silently shifting the criterion by one sample. First tolerance entry can precede qualifying hold start in another session, even though they coincide in the present lane.

## Keyframes and time correspondence

A is the actual t=0 state. C is the first saved sample with |beta|<=epsilon. B is the first argmax of |theta| over [0,t_C], with the endpoint convention declared and the initial state included. D is the state at the final saved time. Use exactly those four elapsed timestamps for the later matched act–coast block, without nearest-success replacement. Identify the sample index behind every pose and timeline marker.

Reconstruct physical links using mechanism_points and the archived geometry. Verify current drawing code matches those archived coefficients before reuse. Keep a fixed camera, scale, apparatus origin, link proportions, arm-aligned hinge axis, and downward/upright convention. The configurations now occupy the exact same piecewise time coordinates as the main traces. Save actual times below the poses and retain A–D correspondence with the exact piecewise time map; do not substitute uniform spacing.

## Future act–coast record requirements

Require the same actual initial state, governing dynamics, torque and arm limits, numerical assessment protocol, hold rule, disturbance assumptions, and full duration or an explicitly recorded early termination. Require timestamped executed states, actual held commands, binary enabled mode, shut-off times, replanning times, selected timing pairs, and failure/termination status. Do not infer mode from a torque deadband. Preserve enabled zero commands, initial activation, and actual zero input throughout coast.

If a matched run terminates early, retain its observed record and terminate the trace visibly at that time. Leave later data absent with a termination annotation; do not extend the terminal state or draw an invented zero torque suffix. A missing or failed act–coast record is not replaced with a different initial condition. Regeneration of controller runs is a separate authorized experiment, not a plotting operation.

## Numerical and visual checks for later production

1. Check manifest hashes for all consumed chunks, initial states, and archived design inputs. Verify 40 chunks, 400 decisions, lane identity, matching configuration and completion metadata; do not rerun the old report script because it rewrites historical outputs.
2. Confirm accepted plans and applied-command decoding, finite samples, 0.001-second sample cadence, 0.1-second command cadence, correct state order, actual t=0, and no epigraph plotted as a control.
3. Reproduce the lane's archived held duration and final-ten-second upright criterion. Report sample-aligned hold start/completion and arm excursions separately. Preserve residual velocities and all failures/violations.
4. Keep all source observations and disable path simplification. Preserve command jumps and extrema. Unwrap only the main displayed pendulum-angle branch; keep all source samples and preserve the physical orientation. Do not average away constraint excursions or merge nearby act–coast events.
5. Match time mapping and ordinate definitions across controller blocks. Preserve no-data whitespace. Check no false branch-cut crossings, clipped peaks, visually interpolated commands, or unequal-camera keyframes.
6. Export 7-inch-wide, 460-point-high double-column page-top vector PDF/SVG plus PowerPoint containing complete high-resolution plot images and independently editable annotations for Canva; use exactly Arial 8/9 points. Inspect package integrity, native annotations and intact plot images, font glyph minimum, plot bounds, and actual-size readability. If Canva autosizes the editing canvas, preserve the local physical-size output and document the scale.
7. Integrate only after plot authorization. Compile the saved main manuscript and inspect its rendered float/page placement before claiming presentation readiness.

## Planned source/output separation

Keep the human plan and interpretation guide in docs/11-inner-MPC/writing/figures/session-record/. Put extraction code, complete numerical CSV/NPZ, selection/transform metadata, exact criteria, and QA under artifacts/session-record/machine-scannables/. Put visual previews and editable presentation under its human-readables/. Copy only involved vector artwork and bilingual caption files into the existing LaTeX project's figures/ directory. No controller implementation, wrapper classes, experiment parameters, GPU invocation, or new numbered development phase is introduced.

## Recorded-energy enhancement

With J the pendulum hinge inertia, A the arm-coordinate inertia and C the coupling inertia at the saved pendulum angle, compute total and relative-swing energy from the same state as the motion traces:

$$
E=\tfrac12 A\omega^2+C\omega\nu+\tfrac12 J\nu^2+V,\quad E_h=\tfrac12 J\nu^2+V,
\qquad s_E=\omega,\quad s_h=-\frac{JC\nu}{AJ-C^2}.
$$

Here V is the downward-referenced potential, and omega and nu are the recorded arm and hinge speeds. A, C, J and V use the archived mechanical model. Normalize both energies by the upright-rest target and both sensitivities by the same downward small-motion frequency. Check energy against energy_components and sensitivities against the governing model input response; do not integrate a new trajectory. Retain full derived arrays and energy_sensitivity.csv.

Implementation budget: load_session and derive_annotations decode and assess the unchanged record; prepare_record recomputes diagnostics; draw_figure renders complete Python plots; main exports paper vectors, intact plot bases and numerical audits. The native exporter places two full images and two independent comparison text boxes. No new classes, wrappers, controller parameters or command-line options. Use compact continuous logic and vectorized arrays, no fallback behavior and no controller runtime modifications.

The main plot image includes the same verified fixed-camera faces, their exact time links, all six trace rows, both adjacent time scales, the enabled rail, legends, ticks and labels. The holding-detail image includes its axes, assessment band and completion annotation. Native text holds only the act–coast heading and pending-record status; controller execution and frequency labels remain in the complete Python base. Do not redraw curve geometry or detach scientific labels in Canva. Full-resolution energy arrays are independently verified against the physical dynamics implementation within absolute tolerance 1e-12. All plotted runs are nominal simulations; n=1 selected session and no aggregate uncertainty is implied.


Current wide layout: width 504 and height 460 paper points. Main views are (left,width,start,end)=(80,270,0,8) and (373,105,8,40). Keyframe centers use these exact maps; their neutral guides stop before the independently scaled late detail. Scalar data are blue; second series are orange dashed, with paired linewidths 0.65/0.5 points. Omega=sqrt(J0*G/(J0*J-B^2)) is verified against the archived natural frequency. The positive inertia determinant and downward-rest linearization determine this scale. Source picture dimensions and time links are retained in artifacts/session-record/machine-scannables/plot_metadata.json; final checks are in validation.json. The LaTeX include is figure* [t] at textwidth.
