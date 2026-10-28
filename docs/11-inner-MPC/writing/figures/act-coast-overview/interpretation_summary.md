# Figure 1: act–coast overview

[Edit the figure in Canva](https://www.canva.com/design/DAHZPzGLBtg/2s_Qwzsfc1g8k594HtjHZg/edit).
The revised Canva design retains separate text, shape regions, filled line segments and arrowheads. Its parallel-MPC label opens in the native text editor with Arial, 9-point type and black text. Timing bars, coasting segments and both critic connections were visually checked after import.

Read panel **a** first. Blue denotes enabled actuation; muted orange denotes zero-input coast. The input and mode share one time axis, and an enabled zero command remains visible. Squares mark planning and replanning; shut-off alone does not require a new observation.

Panel **b** shows how the event is constructed. Different timing pairs feed separate inner MPC problems solved in parallel, then outer MPC compares their minimized scores and selects both times with their compatible input sequence. The actor supplies candidate inputs. The critic estimate is used in the inner terminal cost and is already part of the scores used for outer timing selection; its two arrows do not introduce a second penalty.

[The text inventory and design contract](design_contract.md) explain the purpose of every retained label and identify the prose moved into the caption or Methods. All artwork text is black Arial at two sizes, with blue/orange phases and neutral connectors. Network icons, timing bars and input traces replace explanatory sentences.

The figure represents the **formulated timing-conditioned construction**, not measured act–coast results. Numerical feasibility information is not an admissibility certificate, and the learned continuation estimate does not establish stability or account for all future scheduling charges. The input trace is a schematic with arbitrary amplitudes and durations; no experiment was selected or synthesized.

## Files and editing

- [Editable PowerPoint source](../../../../../artifacts/act-coast-overview/human-readables/figure-1-act-coast-editable.pptx): native shapes, text boxes, filled line segments and arrowheads for Canva import.
- [Manuscript PDF](../../latex-project/figures/act-coast-overview/figure-1-act-coast.pdf): vector output at the existing manuscript's full two-column width.
- [Editable SVG](../../latex-project/figures/act-coast-overview/figure-1-act-coast.svg): live text and vector elements for a vector editor.
- [Caption and placement](../../latex-project/figures/act-coast-overview/figure-1.tex): included by the existing manuscript as a top-positioned double-column float.
- [Preview and Canva proof](../../../../../artifacts/act-coast-overview/human-readables/): visual exports, distinct from the machine-readable geometry and scripts.

Keep the outer timing selection, conditional inner solve and full-event execution scientifically consistent when editing. After changing Canva, export the revised figure as a vector PDF and replace the manuscript figure; the current local PDF does not automatically synchronize with Canva.

## Production milestones and validation

1. **Scientific construction:** match the revised Figure 1 plan and current bi-level subsection. Success means both times are selected outside the conditional input problem, coast is predicted, and continuation is evaluated at replanning.
2. **Editable construction:** create one shared scene, render publication exports, and convert the same scene to a native PowerPoint page. Success means text and shapes remain separate, with no embedded raster picture.
3. **Visual inspection and import:** check the whole figure and both panels, then import to Canva and inspect native editing controls. Success means readable text, clear connectors, and retained diagram structure.
4. **Manuscript placement:** include the PDF at full text width using `figure*` with top placement, then check the existing project with the built-in compiler. Compilation and actual rendered page placement are separate checks.

The implementation uses minimum realization: two scripts, three Python functions (`build_scene`, `render_vector`, `main`), no user-defined classes, wrappers, runtime options or controller parameters. The shared scene stores text, coordinates, fills and paths. Physical width, font and colors are fixed production choices; the symbolic times are controller quantities explained in the caption. No fallback behavior or controller modifications were added.
