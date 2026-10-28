# How to read the revised figure plan

[outline.md](outline.md) now follows the author's five main figures plus optional sixth. Figure 1 is the wide page-top pipeline overview; Figure 2 is the single-column half-page geometry/constraint/goal figure; Figure 3 is single-column energy control and sensitivity analysis. Figure 4 is the wide synchronized session/keyframe comparison, with the act-coast block deliberately blank. Figure 5 is the wide final scenario/baseline resource atlas. Figure 6 optionally covers conventional metrics.

The previous preference for a small benchmark inset and no standalone energy figure is superseded. Keep the act-coast controller central through the overview and resource comparisons, while using the geometry and energy figures to explain its benchmark and mechanical reasoning.

The [source/data contract](machine-scannables/figure_contract.md) separates numerical provenance, energy definitions and metric semantics from the reading plan. Continuous-control records and historical energy assets exist. Sensitivity diagnostics are documented, but their referenced local numerical archive was not found; verify the original source or recompute declared model diagnostics before reuse. Final act-coast and the four requested matched baseline comparisons remain future evidence.

My working interpretation of “total cmp time” is total online controller computation time. The plan distinguishes actuator activations, enabled time, replanning events, nonzero input support and nested computation timers so comparisons do not depend on hidden counting changes.

No figure or blank image was produced, no renderer or experiment ran, no backend was selected, and the LaTeX sources were not edited. This is a Markdown planning update with a Chinese companion. Historical asset numbers do not determine the new manuscript figure numbers.
