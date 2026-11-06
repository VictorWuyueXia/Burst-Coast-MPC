# Figure 3: complete Python plots of energy and sensitivity

This revision replaces the manually projected diagram and the separate preference-plane subplot. The visual argument is that the same recorded motion approaches the target energy while its local input effects alternate between compatible and competing energy-error objectives.

## Panels and purposes

| Panel | Complete scientific plot | Meaning of every mark |
|---|---|---|
| a: Recorded energy path | Adapt the existing Python energy-plane plotting routine to a clear orthographic three-dimensional view, with normal ticks, energy axes and a complete legend. | Blue is the actual saved path in arm kinetic, full pendulum-body kinetic and potential energy; orange is the target-energy plane; an orange diamond is ideal upright rest. Remove the artificial torque-endpoint curve, its work-matching point, the compatibility boundary, and unattached leader labels. |
| b: Sensitivity and competing objectives | Plot the same full historical session directly with Matplotlib axes and a legend. No separate preference-plane panel c remains. | Axes are total/relative-swing torque sensitivities with the same frequency normalization. Blue segments admit a common torque sign for reducing both direct energy-error contributions; orange segments have competing preferences. Classification uses the recorded energy errors, not raw quadrant membership alone. |

Both panels use the already selected slow-downward heuristic session, lane 325, including all 802 retained samples. The pendulum-body kinetic energy in a differs from the hinge-relative swing energy underlying b; define both in the caption. The target plane is an energy level, not an admissible or reachable set. The orange point is an ideal reference, not a claim that the final recorded state is stationary.

## Plotting and editing contract

Use Python-generated complete plots as the scientific bases, including all legends, ticks and labels. Canva receives two high-resolution plot images and is reserved for composition and explanatory highlighting; it must not reconstruct numerical traces, axes or their legends as manually placed shapes. Keep vector PDF/SVG for the paper and 600-dpi PNG plot bases for Canva. Plot changes are made in Python; Canva decorations remain separate.

Preserve the current single-column width, Arial 8/9-point paper text, Morandi blue/orange and neutral gray. Use a view that exposes all three energy directions, retain mathematical normalization, and inspect at final size. Natural swing-energy drift remains separate from torque sensitivity. The conflict coloring is a diagnostic, not an implemented quadrant controller or a guarantee that both objectives can improve together.

## Structure budget and validation

Retain one Python source with three substantive functions: load_record verifies and derives quantities, draw_figure creates the complete plots, and main exports paper vectors and the two complete Canva bases. Reuse the existing energy-plane routine rather than its hand-projected scene. Retain one compact native exporter with two images and no plot-reconstruction shapes. No classes, wrappers, CLI/config parameters, simulation or controller changes.

1. Verify canonical hashes, physical constants, energy identities and saved sensitivity definitions. Success: all quantities reproduce the record and both panels use identical samples.
2. Render and inspect the new viewpoint, visible numerical axes, complete legends, trace extent and font size. Success: no clipped labels or unassigned points/lines and no subplot c.
3. Import plot bases into Canva, preserve current/archived folder organization and update the caption. Success: complete plots are used unchanged, the prior figure is archived, and the existing manuscript compiles.


The current Python plots use descriptive quantity names and scaling in their axes, retain Arial 8/9-point text, and use Morandi blue #7B94A6 and orange #C49D7D for all primary data. Reference lines remain neutral. Figure 3 uses a blue recorded energy path, an orange target-energy plane and the established blue/orange compatibility classes. Figure 4 uses blue arm/input traces, orange pendulum/error traces, paired blue/orange total/swing energy diagnostics, and a pale blue tolerance band. The main pendulum angle stays continuous and centered at downward zero. All scientific labels are part of the Python base; Canva composes intact bases and retains independent explanatory text.
