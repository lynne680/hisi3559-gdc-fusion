# MM5 Three-Modal Fusion Experiments

## Dataset
MM5 aligned multimodal dataset

Used modalities:
- RGB1: visible image
- T8: thermal infrared image
- U8: ultraviolet image

## Model
TinyFusionMaskNet

## Final setting
- mask threshold = 0.70
- base weights   = (0.94, 0.04, 0.02)
- target weights = (0.50, 0.40, 0.10)

## Main results

| Method | EN | SD | AG | SF | MI_SUM |
|---|---:|---:|---:|---:|---:|
| VisibleOnly | 3.7750 | 14.3512 | 0.8554 | 3.4869 | 5.5316 |
| AverageFusion | 6.7446 | 30.9407 | 2.4460 | 6.5467 | 4.6978 |
| FixedWeightedFusion | 6.5804 | 35.4018 | 2.9911 | 9.7144 | 5.1399 |
| OursMaskFusion | 6.0795 | 57.2353 | 3.7840 | 13.8806 | 7.0615 |

## Mask quality at threshold 0.70
- IoU = 0.9297
- Dice = 0.9630
- Precision = 0.9606
- Recall = 0.9665

## Model complexity
- params = 71154
- avg inference time on CPU = 25.035 ms
- fps on CPU = 39.94

## Important files
- tables/main_experiment_metrics.csv
- tables/threshold_sensitivity.csv
- tables/triple_weight_search.csv
- model_info/model_complexity.txt
- images/comparison_000000.jpg
- images/final_fusion_preview_000000.jpg
- images/pred_mask_preview_000000.png
