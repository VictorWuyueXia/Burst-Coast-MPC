# Figure 3: energy request, input authority, and torque preferences

The figure explains why an energy request needs physical-state prediction: a prescribed total-energy plane restricts an energy distribution, and instantaneous torque directions can agree or compete for total and relative swing energies. This is a model-derived explanation of the mechanical teacher, not an act–coast performance experiment or a timing-selection law.

## Composition and text purpose

Use a 3.4-inch single-column composite with one shaded geometric hero panel and two linked diagnostics. Follow Figure 2's projected geometric surfaces and Figure 1's Morandi blue/orange, Arial 8 and 9 points, and black text except labels linked to colored curves. Each mark must convey a quantity, relation, boundary, or correspondence. Avoid decorative frames and duplicated legends.

| Panel | Information and retained text purpose |
|---|---|
| Energy request | Three physical energy axes; upright-rest point; requested-work plane; its exact mechanically compatible subset; the bounded constant-torque endpoint curve and its work-matching intersection. Labels distinguish these sets, rather than imply that one is a reachable set. |
| Torque sensitivity | Total and relative swing power sensitivities along one declared zero-input model segment, with a common positive scale and a zero line. Shared start/end symbols link to the preference plot. |
| Compatible torque signs | Two error-weighted preference axes; common positive/negative torque regions; conflicting regions; the same segment traced through these regions. Zero axes indicate loss of the corresponding direct authority. |

The caption defines all symbols, normalization, constructed examples, and boundaries. Source data retains every integration sample. Rendering uses a declared evenly spaced subset of samples for smooth editable curves, not a selected performance subset.

## Mathematical source and evidence boundary

Use the current undamped model and physics configuration, read-only. Compute a fixed initial state x=(0,pi/3,0.2 Omega,0.6 Omega), where Omega is the model's small-motion natural frequency. Integrate a zero-input segment for T=2pi/Omega. From the same initial state, batch-predict constant torques in [-U,U] for Delta=T/5. The requested work is the existing clipped proportional teacher request with its documented gain. Archive the exact gain, cap, state, durations, and model coefficients in machine-scannables; omit configured values from artwork.

Use E=Ka+Kp+V and Eh=J nu^2/2+V. The latter excludes arm-carried and coupling energy. The two sensitivities are sE=omega and sh=-J C nu/(A J-C^2), evaluated at fixed state. Preferences cE=(E-E*)sE and ch=(Eh-E*)sh are torque coefficients in half-squared energy-error derivatives. Same signs admit a common improving input contribution; drift remains in the complete swing-energy rate.

Normalize energy by E*, sensitivities by Omega, preferences by E* Omega, and time by T. Mechanical compatibility follows from completing the velocity square; it neither enforces arm travel nor determines finite-duration reachability. The endpoint curve is restricted to constant input and one duration. Region signs do not certify full task progress or choose outer MPC times.

## Structure budget and validation milestones

1. Model diagnostics: one diagnostic script, one compute function. Batch all torque predictions; only integration time is sequential. Verify energy/work balance, zero-input conservation, finite-difference sensitivity, exact plane membership, and step-refinement agreement.
2. Drawing: one drawing script with build_scene, render_vector, and main; no new classes, runtime parameters, wrappers, fallback behavior, or controller changes. Fixed palette, physical dimensions, camera, and text sizes. A compact ribbon calculation creates native editable curve shapes.
3. Editable delivery: one JavaScript exporter using the established native-shape scene mapping. Validate one slide, Arial, zero raster pictures, package integrity, and Canva text editing. Inspect complete figure at actual column size.
4. Integration: add only diagnostic definitions needed to interpret the figure, with the control-academic-writing skill; add a bilingual caption and the single-column Methods float, then compile the existing manuscript. Preserve the finite-tolerance task and all controller definitions.
