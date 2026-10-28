# Figure 2: apparatus, hard design limits and ideal goal

The reader should recognize the complete rotary-pendulum apparatus before reading three orthographic views of its operating requirements. The main picture is an original shaded 3D vector illustration. Top, side and front views then explain hard arm travel, free pendulum rotation and exact upright rest. This follows the author's latest instruction and replaces the earlier goal-tolerance cone.

## Retained text and purpose

| Label | Purpose |
|---|---|
| Apparatus; servo base; driven arm; passive hinge; pendulum | Identify the complete mechanism and its actuated and passive joints. |
| +u; torque bound | Show positive shaft torque and its hard design limit. |
| Top; theta; 0; arm bound | Define the fixed horizontal reference and bounded unwrapped arm travel. |
| Side; alpha; 0; pi; free rotation | Explain the passive rod's full rotation and downward/upright orientations without inventing stops. |
| Front; beta = 0; omega = nu = 0 | Show exact upright rest rather than a tolerance region. |

The caption defines all symbols and distinguishes exact design requirements from the implemented numerical penalties and finite-tolerance assessment. The illustration does not claim that soft penalties enforce hard state constraints or establish a certificate.

## Geometry, references and style

Visual references are [Quanser's official apparatus photographs](https://www.quanser.com/products/rotary-inverted-pendulum/) and [workstation brochure](https://www.quanser.com/wp-content/uploads/2017/04/Rotary-Brochure-online.pdf). No photograph, logo or traced artwork is included. Original shaded faces form a generic servo base, vertical drive shaft, horizontal arm, arm-aligned hinge and slender rod. Cosmetic housing dimensions are illustrative; link proportions and motion follow the benchmark's mechanism_points function.

Validate radial/tangential orthogonality, hinge normality, positive rotation, and the model's downward/upright convention. The top view uses a fixed horizontal reference; the side view is normal to the hinge axis; the front view is radial/vertical at ideal rest. Show sharp design limits and an exact upright rod. The illustrative sector does not substitute a configured numerical angle.

Use Figure 1's Morandi blue/orange, black Arial at 8 and 9 points, and neutral gray surface shading. Blue identifies the driven arm; orange identifies the exact upright reference. No extra semantic hue is required.

The IEEE column is 3.4 inches wide. The vector schematic is 3.4 by approximately 3.83 inches; with its caption it occupies roughly half a page. Deliver vector PDF, live-text SVG, PNG preview and native editable PowerPoint shapes for Canva.

## Milestones and structure budget

1. Scientific geometry: numerical assertions confirm the model's rotation basis and downward/upright directions; the solid apparatus retains the correct hinge axis.
2. Minimal construction: two scripts and four Python functions: build_apparatus, build_scene, render_vector and main. The physical configuration is read-only. Camera, surfaces, colors, fonts and paths are fixed drawing values. No new classes, wrappers, runtime options, controller parameters, fallback behavior or controller changes.
3. Visual/editable checks: inspect the 3D apparatus and three orthographic views, text clearance, two font sizes, black text and native shape preservation. Import to Canva and inspect curves and thin lines.
4. Paper integration: update the bilingual caption, include the figure in benchmark Methods and compile the existing source. Keep the human guide separate from geometry and validation records.
