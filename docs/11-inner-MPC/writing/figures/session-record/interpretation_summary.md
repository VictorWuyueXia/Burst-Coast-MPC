# Figure 4: recorded dynamics and control

[Edit Figure 4 in Canva](https://www.canva.com/design/DAHZQqxcBHc/XQywZks33Eh7F-XVUaUlmA/edit).

The figure is drawn directly from the fixed recorded session. Its expanded initial segment shows repeated swing motion, arm excursion, and braking. The complete remaining record shows small upright error with continuing arm movement and nonzero angular speeds. The input staircase is the actual first command executed from each accepted plan; the blue mode rail denotes continuously enabled operation, while the lower rail identifies the periodic feedback cadence.

A–D are physical reconstructions at the actual initial time, the largest arm excursion before first upright entry, first upright-band entry, and the final recorded time. Their labels link the evenly spaced drawing strip to the true time positions on the traces. The camera and link dimensions stay fixed, and each geometry is computed from its saved state. In Canva, each pose and its timestamp form a separate group; enter or ungroup it to edit individual surfaces or text.

- [Publication PDF](../../latex-project/figures/session-record/figure-4-session-record.pdf) and [live-text SVG](../../latex-project/figures/session-record/figure-4-session-record.svg): double-column vector artwork.
- [Editable PowerPoint](../../../../../artifacts/session-record/human-readables/figure-4-session-record-editable.pptx): native text, curve paths, held inputs and configuration faces, with no raster pictures.
- [Separate keyframe assets and previews](../../../../../artifacts/session-record/human-readables/): individual A–D PDF/SVG/PNG configurations and Canva verification screenshots.
- [Caption and placement](../../latex-project/figures/session-record/figure-4.tex): the current-results double-column top float.
- [Complete records, extraction and verification](../../../../../artifacts/session-record/machine-scannables/): full CSV/NPZ observations, source hashes, keyframe states, transforms and drawing checks.

Both time views are linear, with different horizontal scales and no omitted interval. The upright-error ordinate also changes explicitly from beta/pi to beta/epsilon to reveal later holding deviations. Curve breaks indicate angular wrapping, not missing data. The holding tick marks the required duration verified at recorded sample times; it does not prove ideal rest or continuous-time constraint satisfaction.

The act–coast block is intentionally blank. No sparse trajectory, zero-input interval or switching event has been synthesized. The figure does not display batch solve times as per-session latency, or one session as an aggregate resource comparison.

The publication artwork is 7 inches wide, with Arial at 8 and 9 points. Its height was increased from the preliminary budget to preserve legibility and the reserved matched block. Source checksums, archived holding metrics, geometric checks, PDF font checks and native editable-package checks passed. Canva retained the curves and keyframe surfaces; native text editing, Arial, 8-point size and black text were verified. The manuscript compiles; full-paper float and page layout have not been visually reviewed. Canva changes do not automatically update the local PDF.
