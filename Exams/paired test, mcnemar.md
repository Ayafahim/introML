

## S23, Q23) Determine which combination of estimated difference (ˆz), confidence interval (CI) and p-value is the only plausible answer. #histogram

![[Pasted image 20240521182338.png]]



### Question 23 Breakdown

The problem involves a statistical procedure to determine if there is a difference between two regression models, $(M1)$ and $(M2)$, using a paired test with a 5-fold cross-validation setup. We need to determine which combination of estimated difference $(\hat{z})$, confidence interval (CI), and p-value is the plausible answer based on a histogram of paired differences in predictions $(y_{M1} - y_{M2})$.

### Given Information:

- **Histogram of paired differences $(y_{M1} - y_{M2})$.
- **Possible combinations** of $(\hat{z})$, CI, and p-value.

### Goal:

Identify the correct combination of$(\hat{z})$, CI, and p-value.

### Steps to Solution:

1. **Estimate $(\hat{z})$**:
   - The mean of the distribution of paired differences.
   - From the histogram, $(\hat{z})$ appears to be approximately 0.69.

2. **Confidence Interval (CI)**:
   - A CI that includes $(\hat{z})$ and matches the spread of the distribution in the histogram.
   - CI should be centered around $(\hat{z})$.

3. **P-value**:
   - Indicates the significance of the difference.
   - Typically, if CI does not include zero, p-value < 0.05.

### Options Analysis:

A. $(\hat{z} = 0.69), CI = [0.63, 0.75], p-value < 0.05$
- **Estimate $(\hat{z})$**: Matches the histogram.
- **CI**: Centered around 0.69, plausible range.
- **P-value**: < 0.05, indicating significance.

B. $(\hat{z} = -1.05), CI = [-1.29, -0.81], p-value < 0.05$
- **Estimate $(\hat{z})$**: Does not match the histogram.
- **CI**: Does not fit the observed mean.
- **P-value**: < 0.05, but estimate is incorrect.

C. $(\hat{z} = 0.63), CI = [0.57, 0.69], p-value < -0.05$
- **Estimate $\hat{z}$**: Close to histogram but slightly off.
- **CI**: Reasonable, but p-value < -0.05 is invalid (p-value cannot be negative).

D. $(\hat{z} = 0.76), CI = [0.59, 0.76], p-value < 0.05$
- **Estimate $(\hat{z})$**: Slightly higher than observed mean.
- **CI**: Not centered around the estimate.
- **P-value**: < 0.05, but CI is incorrectly centered.

E. Don’t know.
- Not applicable since option A is valid.

### Conclusion:

Option A is correct as it matches the estimated mean $(\hat{z})$, provides a reasonable CI, and the p-value is correctly indicating significance.

---
## S22, Q12)

![[Pasted image 20240521223722.png]]

To solve Question 12, we need to understand McNemar's test and its application in evaluating classifiers. McNemar's test is used for comparing the performance of two classifiers on the same dataset. The test examines the differences in their predictions and determines if there is a statistically significant difference in their performance.

### McNemar's Test

Given two classifiers, McNemar's test uses a contingency table of their prediction results:

|           | Classifier B Correct | Classifier B Incorrect |
|-----------|----------------------|------------------------|
| **Classifier A Correct**   | a                    | b                    |
| **Classifier A Incorrect** | c                    | d                    |

The test statistic is computed as:

$$ \chi^2 = \frac{(b - c)^2}{b + c} $$

McNemar's test evaluates whether the proportions of errors made by the classifiers are significantly different. The test statistic follows a chi-squared distribution with 1 degree of freedom.

### Analyzing the Given Options

We are provided with four sets of estimates, each containing $\hat{\theta}$, a p-value, and a 95% confidence interval (CI). Three of these sets are theoretically impossible when using McNemar's test.

Let's evaluate each option:

1. **Option A**: $\hat{\theta} = 2.05$, p-value < 0.05, CI = [1.45, 2.67]
   - This option seems plausible at first glance. The estimate and CI seem consistent with a significant result (p-value < 0.05).

2. **Option B**: $\hat{\theta} = -0.18$, p-value > 0.05, CI = [0.13, 0.22]
   - This option is incorrect because the CI [0.13, 0.22] does not include the point estimate $-0.18$, which is contradictory.

3. **Option C**: $\hat{\theta} = -0.95$, p-value > 1.05, CI = [-0.98, -0.05]
   - This option is incorrect because the p-value cannot be greater than 1. Also, the CI should not have a negative lower bound for a valid McNemar's test result.

4. **Option D**: $\hat{\theta} = -0.06$, p-value < 0.05, CI = [-0.10, -0.01]
   - This option is theoretically plausible. The point estimate $-0.06$ lies within the CI, and the CI indicates that the result is statistically significant (p-value < 0.05).

Given these evaluations, the only plausible set of estimates when using McNemar's test is:

$$ \boxed{D} $$

### Summary

Option D provides a consistent set of estimates for McNemar's test. The point estimate $\hat{\theta} = -0.06$ lies within the confidence interval $[-0.10, -0.01]$, and the p-value < 0.05 indicates statistical significance, making it the correct answer.