# Segmentation experiments

CubiCasa transferred poorly to FloorPlanCAD, and small fine-tunes did not outperform the structural baseline. These semantic models are not runtime dependencies.

| Model | IoU | F1@3px | Chamfer |
|---|---:|---:|---:|
| Pretrained CubiCasa | 0.113 | 0.391 | 88.83 |
| 44-head fine-tune | 0.117 | 0.408 | 82.20 |
| Binary-512 fine-tune | 0.000 | 0.000 | 1000.00 |

All variants remained below the morphology baseline (`0.182` IoU, `0.477` F1@3px). The semantic model is not required by the PDF pipeline.
