# Final figure reproduction workflow

## Structure and editing scope

Each generated figure has `artifacts/<figure>/src/`, `human-readables/` and `machine-scannables/`. Keep one final Python renderer and one optional editable-PPTX exporter per figure. Figure 4 also keeps the independent archived-record decoder. No new classes, wrappers, CLI options or controller parameters are needed. Retain compact task-oriented files, concise comments and explicit verification; do not add fallback behavior.

## Reproduction milestone

Run all four Python renderers from their package README commands. Success means all final PDF/SVG/PNG outputs are regenerated, source-record assertions pass, and the current figure appearance is preserved. Figures 3–4 must retain complete recorded plots and the empty act-coast comparison.

## Editable-export milestone

Run each optional Node exporter. Success means one canonical PPTX per figure, native editable elements for Figures 1–2, intact final plot tiles for Figures 3–4, and no versioned drafts or inspection sidecars. Verify the Figure 1 action-sequence groups and that embedded scientific images match the final PNG tiles.

## Integration milestone

Compile the existing manuscript with its updated figure copies. Success means the built-in compiler reports success. Compilation alone does not establish final page placement or Canva appearance. Current package inventories and reproduction checks are retained in each package's `machine-scannables/validation.json`.
