# ML From Scratch (C++)

Implementations of core machine learning algorithms built from first principles in C++ — no ML libraries, just the underlying math: cost functions, gradients, and optimization written by hand.

## Why this exists

Built as part of self-directed study following Andrew Ng's ML Specialization, with the goal of understanding the mechanics behind algorithms before relying on high-level frameworks.

## Algorithms

| Algorithm | Status | Notes |
|---|---|---|
| [Linear Regression — One Variable](./linear_regression_one_variable) | ✅ Done | Gradient descent, MSE/MAE |
| [Linear Regression — Multiple Variables](./linear_regression_multiple_variables) | ✅ Done | Z-score normalization, multi-learning-rate comparison |
| [Logistic Regression](./logistic_regression) | ✅ Done | Sigmoid, clamping for numerical stability |
| [KNN](./knn) | 🚧 In progress | 5-feature, 3-class classification |
| Decision Tree | 📋 Planned | |
| Random Forest | 📋 Planned | |
| SVM | 📋 Planned | |

## Structure

Each algorithm folder contains:
- `src/` — C++ source code
- `data/` — dataset used (or description, if not committed)
- `README.md` — implementation details, build/run instructions, and result plots

## Build

```bash
g++ -std=c++17 -O2 src/main.cpp -o main
./main
```
*(adjust per-folder if filenames differ)*

## Author

Dang Duc Viet — HUST, Electronics and Telecommunications Engineering
