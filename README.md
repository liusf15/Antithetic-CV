# Antithetic-CV

Code for reproducing the numerical experiments in
[Cross-Validation with Antithetic Gaussian Randomization](https://arxiv.org/abs/2412.14423) by Sifan Liu, Snigdha Panigrahi, Jake A Soloff.

The paper introduces a method for estimating prediction error via external anti-correlated Gaussian noise. 
For Gaussian distributed data $Y\sim N(\theta, \sigma^2I_n)$, the method constructs $K$ pairs of training-validation data as follows:
```math
Y_{\mathrm{train}}^{(k)} = Y + \sqrt\alpha\omega^{(k)},\quad Y_{\mathrm{test}}^{(k)} = Y - \frac{1}{\sqrt\alpha}\omega^{(k)},
```
where $\omega^{(k)}\sim N(0,\sigma^2 I_n) $ for $1\leq k\leq K$ with the constraint that
```math
\sum_{k=1}^K \omega^{(k)} = 0.
```

`prediction_algorithms.py`: contains prediction algorithms $g$, including isotonic regression, multi-layer perceptron regression, logistic regression, and 2d fused lasso. 

`risk_estimators`: contains risk estimators including the proposed antithetic CV method, the coupled bootstrap estimator, and the classic cross-validation. 

[experiments](experiments/): contains the scripts to reproduce the numerical experiments presented in the paper.


