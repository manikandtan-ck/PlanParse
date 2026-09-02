# Benchmark

## Evaluation setup

The benchmark uses the official FloorPlanCAD test archive.

Three independent frozen drawing cohorts are used. The cohorts do not overlap.

- Vector: 150 drawings across all three cohorts
- Hybrid: 150 drawings across all three cohorts
- Raster: 100 drawings across two cohorts

All detection settings were fixed before final evaluation. No thresholds were tuned on the final cohorts.

Predictions and wall ground truth are evaluated on the same aspect-preserving canvas, with the drawing's long side scaled to 1000 pixels.

## Consolidated results

F1@0.5% is the primary comparison metric. Values below include all evaluated drawings, including drawings with empty wall ground truth.

### Primary metrics

| Method | Drawings | Precision | Recall | F1@0.5% |
|---|---:|---:|---:|---:|
| Vector | 150 | 0.0628 | **0.6787** | 0.3457 |
| Raster | 100 | 0.0572 | 0.5588 | 0.2880 |
| **Hybrid** | **150** | **0.0807** | 0.5434 | **0.3717** |

Values above are means across drawings.

Median F1@0.5% shows the same ordering:

| Method | Median F1@0.5% |
|---|---:|
| Vector | 0.3566 |
| Raster | 0.2855 |
| **Hybrid** | **0.3820** |

Hybrid had the highest median F1@0.5% in each of the three independent Vector/Hybrid evaluation cohorts.

### Full metrics

Values are **mean / median**.

| Metric | Vector | Raster | Hybrid |
|---|---:|---:|---:|
| Drawings | 150 | 100 | 150 |
| Non-empty GT | 133 | 87 | 133 |
| IoU | 0.0614 / 0.0419 | 0.0552 / 0.0331 | **0.0775 / 0.0501** |
| Precision | 0.0628 / 0.0422 | 0.0572 / 0.0339 | **0.0807 / 0.0524** |
| Recall | **0.6787 / 0.7842** | 0.5588 / 0.6942 | 0.5434 / 0.5929 |
| Pixel F1 | 0.1015 / 0.0805 | 0.0861 / 0.0641 | **0.1238 / 0.0955** |
| F1@0.3% | 0.2694 / 0.2636 | 0.2254 / 0.2188 | **0.2981 / 0.2909** |
| F1@0.5% | 0.3457 / 0.3566 | 0.2880 / 0.2855 | **0.3717 / 0.3820** |
| F1@1.0% | 0.4694 / 0.5167 | 0.3949 / 0.4203 | **0.4897 / 0.5390** |
| Normalized Chamfer | 0.1406 / 0.0260 | 0.1912 / 0.0395 | 0.1417 / **0.0258** |

![Replicated PDF-mode benchmark across three frozen cohorts.](../assets/benchmark_replication.png)

## Results across independent cohorts

Hybrid and Vector were compared independently on each of the three frozen cohorts.

| Cohort | Hybrid better | Vector better | Tied |
|---|---:|---:|---:|
| Frozen cohort A | 28 | 15 | 7 |
| Frozen cohort B | 24 | 14 | 12 |
| Frozen cohort C | 30 | 15 | 5 |

Hybrid had the higher median F1@0.5% in all three cohorts.

The result is consistent across independently frozen sets rather than being driven by a single sample group.

## Interpretation

**Vector** retains more wall pixels and therefore has the highest recall. It also accepts more non-wall line structures.

**Hybrid** uses rendered-image evidence to reject some of those candidates. This lowers recall but improves precision and produces the highest replicated F1@0.5%.

**Raster** does not depend on native PDF geometry and remains useful for image-based or scanned plans, but it is more affected by visual clutter.

Based on these results, Hybrid is the current default when usable PDF drawing geometry is available.

## Benchmark history

An earlier five-drawing comparison was used during initial development. It has been replaced by the current multi-cohort evaluation because the larger benchmark provides a more reliable comparison and uses validated alignment between the source drawing, generated PDF and wall ground truth.

The older five-drawing values are not used in the current reported results.

Machine-readable consolidated values are available in [`results.csv`](results.csv).

`vector_benchmark/results.csv` uses the same aggregate schema. Per-drawing rows were not reconstructed from aggregate results.
