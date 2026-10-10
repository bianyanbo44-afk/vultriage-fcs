# TRACE aggregate evidence dictionary

The original release has 18 frozen members, identified by `analysis/public_analysis_manifest.json`. The additional release has 48 aggregate/receipt members, identified by `analysis/neural_validation_v3/public_analysis_manifest.json`. These counts include manifests and fit receipts as declared by each release; they do not count independent experiments. `display_additions/manifest.json` separately binds the post-review compact displays without changing either original release.

## Additional evidence files

| File or group | Contents and unit |
|---|---|
| `*/project_seed_metrics.csv` | Project, actual fit, budget, clip and policy endpoints; fractions and cost per function |
| `*/project_seed_averages.csv` | Within-project averages across the actual fitted seeds, with defined-value and policy-availability rules retained |
| `*/paired_project_summaries.csv` | Weighted-minus-comparator project contrasts, both declared rosters, paired medians and pointwise project-bootstrap intervals |
| `*/domain_supported_paired_summaries.csv` | The label-free fixed domain-available secondary roster |
| `*/domain_fit_availability.csv` | Grouped domain-fit availability for each project and actual fit |
| `*/label_free_decision_counts.csv` | Aggregate assigned-action counts, not per-function decisions |
| `project_allocation_structure.csv` | Primary project action counts/fractions after actual-fit averaging, disagreement and clipping/infinity diagnostics |
| `allocation_structure_summary.csv` | Project medians and exact-action-equality counts for the complete and supported rosters |
| `training/*.json` | Seven selected-fit identities, checkpoint/source/configuration hashes and reported training settings |
| `display_additions/action_fractions.csv` | Primary marginal project medians of assigned-action fractions and median paired vulnerable-action changes |
| `display_additions/allocation_sensitivity.csv` | Every supported fine-tuned allocation-miss cell, including intervals and direction counts |
| `display_additions/cost_grid_summary.csv` | Every 27-cell free-volume cost family, median ranges and interval-direction counts |

## Fields and denominators

| Field or concept | Definition |
|---|---|
| `dataset`, `project`, `detector`, `seed` | Dataset/project/fitted-object identity. Three fine-tuned and three hashing seeds are actual fits; the frozen reference has one head. Fitted objects are reused across collections. |
| `alpha_v`, `alpha_s`, `clip` | Vulnerable/safe requested set-miscoverage budgets and domain-odds clipping limit. The additional primary setting is 0.10, 0.20 and 20. |
| `n`, `auto_safe`, `auto_vulnerable`, `review_count` | Input and assigned-action counts. Project averages can be fractional because actual-fit counts are averaged. |
| Automatic-miss rate | Automatic-safe vulnerable functions divided by all vulnerable functions in that project |
| Automatic false-alert rate | Automatic-vulnerable safe functions divided by all safe functions in that project |
| Selective error | Automatic errors divided by automated functions; undefined when automation count is zero |
| Set miscoverage | True-label exclusion from the prediction set, including empty sets that receive review; undefined when weighted sets cannot be constructed |
| `cost_10`, `cost_50` | Cost per function with false-alert cost 1, review cost 0.1, correct review and the named missed-vulnerability cost |
| `median_difference` | Median of within-project weighted-minus-comparator differences after actual-fit averaging; not a difference of marginal medians |
| `ci_lower`, `ci_upper` | Pointwise percentile bounds from the specified 10,000-draw project bootstrap, conditional on the fitted models |
| `reductions`, `increases`, `ties` | Negative, positive and tied project differences under the declared endpoint tolerance |
| `cohort`, `status`, `defined_projects` | Explicit roster, availability/failure status and number of defined endpoints. Missing endpoints are retained rather than silently removed. |
| Clipping/infinity fractions | Label-free target-point diagnostics for a named class threshold; they do not measure density-ratio accuracy or condition on true target class |

Blank CSV cells represent undefined quantities, not zero. Complete review has zero observable automatic errors and cost 0.1, while conditional selective error and unavailable fitted-set miscoverage remain undefined. Zero-width median intervals can arise from discrete tied differences and do not establish equivalence.

The additional DiverseVul and MegaVul collections have 38 and six project groups, with four names shared. Their 40 distinct names are not an assertion of independent components, commits, CVEs or pretraining inputs. The collections are not pooled. Fixed budgets and clips reuse projects and fits and are not additional replications.

Software versions and the distinction between the original control environment and the later dedicated validation environment are recorded in `reproducibility/validation_runtime_v3.json` and the companion requirements files. Full experimental replay requires independently obtained hash-matched inputs and the historical screening artifacts; the aggregate reconstruction does not.
