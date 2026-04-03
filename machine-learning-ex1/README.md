# Machine Learning - Exercise 1: Linear Regression

Stanford Coursera course by Andrew Ng.

## Overview

Two-part exercise implementing linear regression from scratch in MATLAB/Octave:

1. **Part 1:** Single variable - predict food truck profit from city population
2. **Part 2:** Multi-variable - predict house prices from size and bedrooms

## Files

### Data
- `ex1data1.txt` - Single variable training data (population, profit)
- `ex1data2.txt` - Multi-variable training data (sq-ft, bedrooms, price)

### Scripts
- `ex1.m` - Part 1 main script
- `ex1_multi.m` - Part 2 main script

### Functions to Implement
| File | Description |
|------|-------------|
| `warmUpExercise.m` | Return 5x5 identity matrix |
| `plotData.m` | Scatter plot of training data |
| `computeCost.m` | Cost function: J(θ) = (1/2m) Σ(prediction - y)² |
| `gradientDescent.m` | Gradient descent for single variable |
| `featureNormalize.m` | Z-score normalization: (x - μ) / σ |
| `computeCostMulti.m` | Cost function for multi-variable case |
| `gradientDescentMulti.m` | Gradient descent for multiple variables |
| `normalEqn.m` | Closed-form: θ = (XᵀX)⁻¹Xᵀy |

### Supporting
- `lib/` - Helper library functions
- `submit.m` - Submission script
- `ex1.pdf` - Full assignment description

## Running

```matlab
% Part 1
ex1

% Part 2
ex1_multi
```

In Octave:
```octave
octave --no-gui
```

## Theory

**Gradient Descent:** Iteratively minimize cost function J(θ) by updating:
```
θ := θ - α * (1/m) * Xᵀ(Xθ - y)
```

**Normal Equation:** Closed-form solution (no need for learning rate or iteration):
```
θ = pinv(X' * X) * X' * y
```

**Feature Normalization:** Required for multi-variable case since features have different scales. Normalize to mean=0, std=1.

## Results

| Metric | Part 1 | Part 2 |
|--------|--------|--------|
| θ from gradient descent | [-3.63, 1.17] | ~[334302, 100165, 3670] |
| Prediction (35K pop) | $4,518 | - |
| Prediction (1650 sq-ft, 3 br) | - | ~$293,000 |
| Prediction via normal eqn | - | ~$293,000 |
