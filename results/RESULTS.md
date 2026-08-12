# Final Reported Results

## Validation Set — 920 images

| Model | Accuracy |
|---|---:|
| CNN | 94.78% |
| Random Forest | 95.98% |
| SVM | 93.15% |
| KNN | 97.07% |
| Logistic Regression | 92.07% |
| **Weighted Ensemble** | **97.50%** |

Ensemble confusion matrix: `[[402, 15], [8, 495]]`.

## Separate Unseen Set — 600 images

| Model | Accuracy |
|---|---:|
| CNN | 91.67% |
| Random Forest | 90.67% |
| SVM | 90.83% |
| KNN | 89.67% |
| Logistic Regression | 88.83% |
| **Weighted Ensemble** | **97.17%** |

Ensemble confusion matrix: `[[296, 4], [13, 287]]`.

These values are taken from the executed outputs of the supplied thesis notebook for the saved hybrid pipeline.
