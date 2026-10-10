# TRACE v4 public research evidence

TRACE evaluates risk, automatic-label allocation and cost under project shift. This increment adds analyst-capacity comparisons, delayed reviewed-label simulations, known-density diagnostics, the executed official PROM conformal component, and an independently trained CodeT5 encoder reference.

With review limited to 20% of each project's functions, mandatory TRACE leaves unresolved work in all six MegaVul projects and most of the 38 DiverseVul projects; the capped source-cost and MSP policies complete every roster. Policy rankings depend on capacity: for fine-tuned CodeBERT on DiverseVul, source-cost selection minus MSP has median cost difference -0.0569 at 20% capacity but +0.0275 when the full roster is reviewable, while the MegaVul difference at 20% is +0.00926. Delayed reviewed-label updates provide no consistent improvement. At 16 slots and a three-batch delay, the median DiverseVul project has only three batches and no feedback arrives during its run; only 11–12 of 38 projects produce a source-calibration update for any seed. MegaVul offers longer histories but mixed cost changes. These are known-outcome benchmark simulations under miss10, false-alert1, review0.1 and correct-review assumptions in illustrative cost units; MegaVul follows recorded author dates and DiverseVul a synthetic order, with no dated source-before-target validation.

The v3 public evidence, source, generated tables and figures retain their original bytes. New aggregate-only packets are under `public_results/v4/`; each preserves its own source manifest. The three final v4 figures, four source CSVs, plotting source/helper, contract, manifest and visual/geometry QA are under `public_results/kbs-validation-v4/figure-data-v4/`. Corrected CodeT5 uses attempt-2 and the v4.1 protocol. The older prepare/audit filenames are read-only helper dependencies, not the discarded training experiment.

The full repository archive provides the complete checked staging tree; the increment archive supplies only additions to the published v3 research artifact. `TRACE_V4_RELEASE_MANIFEST.json` records every payload hash and the originating module manifests. The inherited `ARCHIVE_MANIFEST.json` describes the v3 subset only.

Original v3 aggregate reconstruction remains available through its unchanged README and checkers. Capacity tables retain every fixed scenario and project, with ordinary cost undefined when work is unresolved. Ratio tables summarize independent simulation repetitions. PROM and CodeT5 provide aggregate reconstruction entrypoints; use the module README for its input paths and numerical-library versions. Full detector/row replay additionally requires independently acquired benchmark/model inputs described by the protocols.

Packaging checks and archive verification are separate from experiments:

```
python scripts/build_trace_v4_public_release.py verify --tree .
```

No benchmark functions, per-function probabilities/actions/events, model weights, token caches, label vaults, author Word/photos, mock reviews or numerical vendor installations are supplied. PROM's original root and source licence notices, attribution and separate modification diffs are retained; this increment assigns no replacement software/data licence. Publication and manuscript submission are separate author actions.
