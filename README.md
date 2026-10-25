# Linear Regression with the Normal Equation

Linear regression implemented from scratch with NumPy, using the normal equation, and applied to the California Housing dataset.

## What this project does

- Loads the California Housing data from scikit-learn (8 features, target: median house value in units of $100,000).
- Adds a bias column and splits the data 80% train / 20% test.
- Solves for the weights directly with the normal equation: theta = (XᵀX)⁻¹ Xᵀy.
- Predicts on the test set and computes RMSE and R² by hand (no sklearn metrics).

## Results (test set)

| Metric | Value |
|---|---|
| RMSE | 0.727 (about $72,700) |
| R² | 0.594 |

## Limitations

- A linear model explains only about 59% of the variance in house prices. The relationship with features such as latitude and longitude is not linear.
- The features are not scaled. This does not affect the normal equation, but it makes the weights hard to compare with each other.
- The code uses an explicit matrix inverse (`np.linalg.inv`). `np.linalg.solve` or `lstsq` is more numerically stable.

## How to run

Install the libraries, then open the notebook:

```
pip install numpy pandas scikit-learn jupyter
jupyter notebook house_pricing.ipynb
```