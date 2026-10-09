# Reading the controller paper

Open [main.tex](main.tex) in the native LaTeX editor for the English manuscript and its PDF preview. [main-CN.tex](main-CN.tex) is the Chinese companion. The English manuscript uses the project’s `ieeeconf.cls` and the files under `figures/`; the Chinese companion retains its earlier standalone layout. The [Overleaf upload bundle](machine-scannables/overleaf-project.zip) contains the English manuscript and all its LaTeX dependencies. They follow the accepted [21-paragraph plan](paragraph_plan.md): three introduction paragraphs, problem formulation, two method subsections, and one results paragraph supported by tables and a six-panel atlas.

## What the draft explains

The controller requests motor work from three physical energy components. That request defines a plane in energy space. At a fixed physical state and action duration, varying torque traces an endpoint-energy curve. Their intersections supply work-matching candidates, which are scored together with grid torques. The final choice trades potential energy, work mismatch, and a soft arm penalty.

The manuscript connects this construction to the Lagrangian, physical-state equations, body-energy partition, and power balance. It distinguishes the pendulum phase needed to predict energy transfer from the absolute arm position needed to assess excursion. Holding and intervention metrics are formulated as prospective evaluation criteria.

## How to read the figures

- **Method figure:** the mechanism and an actual one-action decoder diagnostic at a recorded state. Circles mark work matches; the star is the logged selected action. The shaded plane represents a budget, not a reachable set.
- **Atlas A–B:** all 4,096 randomized validation episodes. All captured; six moving starts crossed the arm envelope. Fixed probes are counted separately.
- **Atlas C–E:** selected downward replays show angles, applied torque, and unweighted physical energies. Their 11.12 s and 16.92 s durations describe selected cases; the full downward population has an 11.13 s median.
- **Atlas F:** two recorded crossings distinguish an available bounded tested alternative from the absence of any bounded tested action at the preceding decision.

My assessment is that the energy-budget and phase-decoding argument is coherent and faithful to the current code. The strongest present evidence is nominal angle capture. A publication study still needs continued holding trajectories and a meaningful controller comparison before making stability, robustness, or intervention-efficiency claims.

## Completed checks and remaining review

| Milestone | Check | Status |
| --- | --- | --- |
| Mathematical draft | Equations checked against the physical model and decoder; all 21 paragraph roles retained. | Complete |
| Recorded figures | Full validation counts reproduced; selected CSV hashes verified; diagnostic endpoint matches the logged prediction within an absolute tolerance of 10⁻¹² J. | Complete |
| Writing passes | Nature writing and control writing, followed by Nature polishing and an academic-humanizer audit and revision. The last prose pass preserved all numerical tokens, displayed equations, and citations. | Complete |
| LaTeX compilation | Earlier standalone versions compiled. The current English project uses external files, which the built-in compiler cannot resolve. | Overleaf compilation pending |
| Rendered page inspection | Computer-use access to Codex is disallowed, and the compiler tool does not export PDF files. | Layout remains visually unverified |

For advisor review, start with the holding target in Section II and the candidate-ranking rule in Section III.B. These determine the next experimental questions. The current draft makes no claim that capture is sustained balancing or that the finite arm penalty guarantees containment.

The [machine-scannables folder](machine-scannables) holds the source map, numerical figure tables, provenance checks, and writing audit. The preparation script is [prepare_controller_paper.py](../../scripts/prepare_controller_paper.py). The eight completed videos remain in the replay bundle's [human-readables folder](../../artifacts/rotary_pendulum/experiment-results/single-constant-torque-action-controller/portable-animation-records/human-readables/interpretation_summary.md).
