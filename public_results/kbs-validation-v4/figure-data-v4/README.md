# TRACE v4 figure evidence

The three final figures and their four source CSVs retain the complete DiverseVul and MegaVul project rosters. The workflow is conceptual; the two quantitative figures use the fixed primary settings declared in `figure_contract_v4.json`.

`figure_manifest_v4.json` binds the plotted CSVs, final PDF/SVG/PNG/TIFF exports, source preflight, physical geometry, glyph audit, alignment/collision reports and visual review. `visual_review_v4.json` documents inspection of the exact final PDF/PNG exports.

The plotting source is available both in `source/` and at repository `src/make_kbs_strengthening_figures_v4.py`. Its unchanged v3 helper, `src/make_neural_validation_figures.py`, is included in `source/` for inspection. Both source hashes are recorded in the original figure manifest.

The source reads the checked public aggregate packet through `--aggregate-dir`. Its manifest records both the original statistical analysis binding and the additional public-capacity-manifest binding. The complete four plotted CSVs also support direct data inspection.

To render from the public repository tree, supply the external figure QA scripts directory:

```
python src/make_kbs_strengthening_figures_v4.py --aggregate-dir public_results/v4/capacity_feedback --audit-scripts PATH_TO_QA_SCRIPTS
```

The checked contract is also copied to `paper/neurocomputing_revision/figures/v4/figure_contract_v4.json`, the renderer's expected runtime path. The renderer writes new exports into that directory. The external `--audit-scripts` directory supplies `validate_figure.py`, `audit_panel_alignment.py`, `audit_pdf_text.py` and `audit_figure_collisions.py`; the skill/QA runtime is an explicit external dependency. Packaging verifies the saved export hashes and recorded QA gates without running experiments or rendering figures.
