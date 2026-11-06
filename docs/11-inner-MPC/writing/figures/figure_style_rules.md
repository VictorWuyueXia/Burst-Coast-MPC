# Figure style rules from the author's examples

These rules supplement the established act–coast figure style and apply to later plotting and refinement. The examples were visually inspected from the seven supplied files in `/Users/vic/Downloads/example_figures`, including both one-page PDFs. They are composition references, not sources of scientific data or artwork to copy.

## What the examples do well

| Reference | Useful visual grammar | Application to our figures |
|---|---|---|
| ICRA workflow PNG | A broad overview precedes detailed explanations; stacked cards encode a trajectory set; miniature action/state sequences explain what each candidate contains. | Show the act–coast concept before the algorithm. Draw a stack of conditional inner optimization problems, each representing one timing pair. |
| Two encoder/classifier PNGs | Repeated symbols retain their meaning across alternatives; arrows connect visible representations rather than paragraphs. | Use one small timing diagram repeatedly, with the same blue/orange meanings, and give each block a visible input/output. |
| Sensory development PNG | Aligned metric panels, consistent scales and category order make comparisons direct. | Keep common time axes, normalization and method ordering when comparing sessions or baselines. |
| WildFusion poster | A compact input–model–output diagram is grounded by physical examples; sections have clear hierarchy. | Use physical illustrations to connect the controller to the task without allowing the benchmark to dominate the method. |
| ICRA PDF poster | Method sketches and results occupy complementary areas; related example cases use repeated layouts. | Separate mechanism from measured evidence, and use repeated miniatures for genuinely comparable cases. |
| fr3-sonopet PDF | Enclosing regions identify functional subsystems; concrete sensor/robot views explain abstract interfaces. | Group elements that have one scientific role and keep their internal structure editable. |

## Rules for later work

1. Begin with one visual statement. Readers should understand the mechanism before reading the caption. Give each panel a different explanatory job.
2. Use geometric structure to carry information: stacks mean a collection, aligned repeated cards mean comparable problems, brackets mean durations, and arrows mean a specified information or execution dependency. Decorative icons are unnecessary.
3. Make each block a miniature explanation, not just a named rectangle. Show the timing pair and optimized input sequence inside an inner-MPC card; a neural icon should indicate a learned component rather than imply a particular network architecture.
4. Reuse a small visual vocabulary across the entire paper. Keep physical poses, action sequences, candidate cards and target markers consistent. Repetition is valuable when it communicates multiplicity or comparison.
5. Increase information per meaningful object rather than object count. A card stack can explain batch computation with a few faces; a dense staircase can explain held discrete commands with one grouped editable sequence.
6. Establish hierarchy through scale, alignment and whitespace. Use one dominant mechanism panel and subordinate workflow or evidence; align block edges and route connections through reserved gaps. Main flow proceeds rightward and downward with one arrow style.
7. Put short labels next to their objects. Keep only names, quantities and distinctions that the geometry cannot explain; place definitions, qualifications and derivations in the caption or prose.
8. Keep boundaries subordinate. Use pale fills and fine outlines for conceptual groups, stronger strokes for actions and essential connections, and enough separation to distinguish dense content at final paper size.
9. Compare like with like. Share axes, normalization, camera, scales and category order where the scientific comparison requires them. A composite figure should answer several linked questions, not repeat the same evidence.
10. Preserve evidence status. Symbolic sequences are illustrations, recorded trajectories are measurements from a specified run, and an unspecified selection rule remains an explicit placeholder. A stack is not a measured speed-up claim.
11. Preserve human editability at the right level: group related line segments into a movable action sequence while leaving labels and workflow blocks independently editable. Keep the current design in `act-coast` and superseded Canva versions in `_archive`.
12. Review the assembled figure at publication size. Every mark must remain legible and have a stated purpose; automated font or package checks supplement visual review.

## Established constraints retained

Use at most two main hues and two accents; our current figures need only Morandi blue #7B94A6, orange #C49D7D, and neutral grays. Keep Arial only, with 8-point supporting text and 9-point headings, black text except direct color-linked concept labels. Avoid red/green comparisons. Retain Figure 2's geometric illustration style and Figure 1's semantic colors. Paper figures do not inherit poster-scale typography, extra colors, elaborate borders or unreadable embedded miniatures from the examples.

Figure 1 remains a wide double-column page-top overview. The rotary pendulum remains a benchmark. Mathematical captions define every displayed symbol and distinguish the formulated timing controller from evaluated continuous MPC+RL performance.

For numerical scientific plots, use complete Python-generated plots with their axes, ticks, legends and labels as the figure bases. Canva composes those intact bases and adds separate explanatory highlights; do not reconstruct data curves or axes manually. Revise the numerical plot in Python and replace its complete base. Native editable schematic shapes remain appropriate for conceptual workflows such as Figure 1.


Scientific axes must name the physical quantity and its normalization in Python rather than displaying an unexplained symbol. Retain necessary notation only after the quantity name, such as pendulum angle / pi or energy-rate sensitivity / Omega. Use Morandi blue/orange for primary data curves, not grayscale. Black text and neutral reference/grid lines remain subordinate. For time-series plots, use blue for the first or sole data series and orange for a second, with legends naming both and thin strokes where curves overlap. This is a row-local series order, not a global component identity. Retain the continuous pendulum-angle branch centered at downward zero. Recheck label clearance at final column width after adding descriptive words.


## Authoritative figure formats and planning checks

| Figure | Purpose and required format |
|---|---|
| 1 | Act–coast concept first, then the role of each component in an engineering pipeline; wide double-column page-top overview. |
| 2 | Rotary-pendulum geometry, operating constraints and ideal goal; single-column, about half a page high. |
| 3 | Energy control in energy space, sensitivity curves and their regions; single-column. |
| 4 | One-session system and controller time record with physical keyframes linked to actual times; continuous control populated, matched sparse act–coast left blank; wide double-column page-top. |
| 5 | Wide double-column page-top performance atlas across declared scenarios, comparing bang–bang, event-triggered MPC, maximum hands-off control and mixed temporal resolution MPC on activation count, enabled/control time and total online computation time. Do not populate absent comparisons. |
| 6, optional | Conventional control-performance comparison with the baselines, only if it answers a question not already resolved by Figure 5. |

Before drawing or revising, iterate through every topic and explicitly write: what question it answers; whether that question is necessary for the figure's argument; the clearest representation; and its placement relative to the other topics. Avoid multiple objects conveying one weak point or sparse content conveying little. Use few meaningful elements to communicate several necessary pieces of information, and remove decorations that duplicate axes, legends or annotations.

Retain at most two main colors and two accents, one font and at most two font sizes. Use black text unless color directly identifies a plotted concept, Morandi blue/orange rather than red/green, and complete Python plots with descriptive labels. Distinguish scientific color classes from the ordinary first/second-series convention. Figure 4's width and linked-keyframe requirements supersede its former single-column layout.
