
## F23, Q21)

Consider the synthetic 1D dataset with four training observations:

$[ X = \{-3, -1, 5, 6\} ]$

A standard kernel density estimator (KDE) with $\sigma^2 = 0.5$ is used to determine if test observations are anomalous based on their estimated densities. An observation will be flagged as an anomaly if $p(x) < 0.015$. We test two new observations located at $x = -4$ and $x = 2$.

Determine which one of the following statements is correct.

A. Both test observations are anomalies.

B. The observation at $x = -4$ is not an anomaly but the observation at $x = 2$ is an anomaly.

C. The observation at $x = -4$ is an anomaly but the observation at $x = 2$ is not an anomaly.

D. Neither of the test observations are anomalies.

### Breakdown:

1. **Kernel Density Estimation (KDE):**
   - KDE is a non-parametric way to estimate the probability density function (PDF) of a random variable.
   - The KDE formula for a Gaussian kernel is:
 $$
     
     \hat{p}(x) = \frac{1}{n \sqrt{2 \pi \sigma^2}} \sum_{i=1}^n \exp \left( -\frac{(x - x_i)^2}{2 \sigma^2} \right)
     
$$
   - Here, $n$ is the number of observations, $\sigma^2$ is the bandwidth parameter, and $x_i$ are the training observations.

2. **Calculate KDE for Test Observations:**

   - Given $\sigma^2 = 0.5$, $\sigma = \sqrt{0.5} = \sqrt{0.5} \approx 0.707$.
   - Training observations: $X = \{-3, -1, 5, 6\}$.

3. **For $x = -4$:**
 $$
   \hat{p}(-4) = \frac{1}{4 \sqrt{2 \pi \cdot 0.5}} \left( \exp \left( -\frac{(-4 - (-3))^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(-4 - (-1))^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(-4 - 5)^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(-4 - 6)^2}{2 \cdot 0.5} \right) \right)
   $$
$$
   \hat{p}(-4) = \frac{1}{4 \cdot \sqrt{\pi}} \left( \exp \left( -2 \right) + \exp \left( -\frac{9}{0.5} \right) + \exp \left( -\frac{81}{0.5} \right) + \exp \left( -\frac{100}{0.5} \right) \right)
$$
$$
   \hat{p}(-4) \approx \frac{1}{4 \cdot \sqrt{\pi}} \left( \exp \left( -2 \right) + \exp \left( -18 \right) + \exp \left( -162 \right) + \exp \left( -200 \right) \right)
  $$
$$
   \hat{p}(-4) \approx \frac{1}{4 \cdot \sqrt{\pi}} \left( 0.1353 + \text{negligible terms} \right)
$$
$$
   \hat{p}(-4) \approx \frac{0.1353}{4 \cdot \sqrt{\pi}} \approx 0.019
$$

4. **For $x = 2$:**
$$
   \hat{p}(2) = \frac{1}{4 \sqrt{2 \pi \cdot 0.5}} \left( \exp \left( -\frac{(2 - (-3))^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(2 - (-1))^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(2 - 5)^2}{2 \cdot 0.5} \right) + \exp \left( -\frac{(2 - 6)^2}{2 \cdot 0.5} \right) \right)$$
$$
   \hat{p}(2) = \frac{1}{4 \cdot \sqrt{\pi}} \left( \exp \left( -\frac{25}{1} \right) + \exp \left( -\frac{9}{1} \right) + \exp \left( -\frac{9}{1} \right) + \exp \left( -\frac{16}{1} \right) \right)$$
$$
   \hat{p}(2) \approx \frac{1}{4 \cdot \sqrt{\pi}} \left( \text{negligible terms} + \exp \left( -9 \right) + \exp \left( -9 \right) + \text{negligible terms} \right)
$$
$$
   \hat{p}(2) \approx \frac{1}{4 \cdot \sqrt{\pi}} \left( 2 \cdot \exp \left( -9 \right) \right)
$$
$$
   \hat{p}(2) \approx \frac{2 \cdot 0.000123}{4 \cdot \sqrt{\pi}} \approx \frac{0.000246}{4 \cdot \sqrt{\pi}} \approx 0.00004
$$

### Conclusion:
- For $x = -4$, $\hat{p}(-4) \approx 0.019$.
- For $x = 2$, $\hat{p}(2) \approx 0.00004$.

Given that an observation is flagged as an anomaly if $p(x) < 0.015$:

- The observation at $x = -4$ is not an anomaly because $\hat{p}(-4) \approx 0.019$ is greater than 0.015.
- The observation at $x = 2$ is an anomaly because $\hat{p}(2) \approx 0.00004$ is less than 0.015.

The correct answer is:
B. The observation at $x = -4$ is not an anomaly but the observation at $x = 2$ is an anomaly.

![[Pasted image 20240521012728.png]]

## Scripts (2 versions)

```python
import numpy as np  
  
# Define the KDE function for Gaussian kernel  
def kde_gaussian(test_points, data, bandwidth):  
    n = len(data)  
    factor = 1 / (n * np.sqrt(2 * np.pi * bandwidth))  
    # Ensure proper broadcasting by reshaping the data  
    kde_values = factor * np.sum(np.exp(-0.5 * ((test_points - data.T) ** 2) / bandwidth), axis=1)  
    return kde_values  
  
# Function to check if observations are anomalies  
def check_anomalies(test_points, data, bandwidth, threshold):  
    densities = kde_gaussian(test_points, data, bandwidth)  
    anomalies = densities < threshold  
    return densities, anomalies  
  
# Define the dataset and parameters  
data = np.array([-3, -1, 5, 6])  
sigma = 0.5  
threshold = 0.015  
test_points = np.array([-4, 2])  
  
# Calculate densities and check for anomalies  
densities, anomalies = check_anomalies(test_points.reshape(-1, 1), data.reshape(-1, 1), sigma, threshold)  
  
# Print the results  
for i, x in enumerate(test_points):  
    print(f"Density at x = {x}: {densities[i]:.6f}, Anomaly: {anomalies[i]}")  
  
# Easily modifiable sections  
# To change the dataset, modify the 'data' variable.  
# To change the test points, modify the 'test_points' variable.  
# To change the bandwidth, modify the 'bandwidth' variable.  
# To change the anomaly threshold, modify the 'threshold' variable.
```

```python
import numpy as np # type: ignore  
  
# Training observations  
#TODO: update  
X = np.array([-3, -1, 5, 6])  
  
# Test observations  
#TODO: update  
test_observations = np.array([-4, 2])  
  
# Standard deviation  
σ = np.sqrt(0.5)  
  
# Gaussian kernel function  
def K(x):  
    return (1 / np.sqrt(2 * np.pi * σ**2)) * np.exp(-x**2 / (2 * σ**2))  
  
# Calculate KDE for each test observation  
for x_test in test_observations:  
    KDE = np.mean(K(x_test - X))  
    print(f"KDE for x = {x_test}: {KDE}")  
      
    # Check if the KDE is less than 0.015  
    #TODO: update  
    if KDE < 0.015:  
        print(f"The observation at x = {x_test} is an anomaly.")  
    else:  
        print(f"The observation at x = {x_test} is not an anomaly.")
```
--- 
## S19, Q25) Which of the curves in Figure 13 shows the LOO estimate of the generalization error E(σ)?

![[Pasted image 20240521144543.png]]![[Pasted image 20240521144608.png]]

## Solution 
The dataset consists of 4 observations: $\{3.918, -6.35, -2.677, -3.003\}$.

### Kernel Density Estimation (KDE):
Kernel Density Estimation is a non-parametric way to estimate the probability density function of a random variable. The kernel width $\sigma$ (or bandwidth) controls the smoothness of the estimated density.

### Leave-One-Out Cross-Validation:
LOO cross-validation is used to estimate the performance of a model by removing one observation at a time, fitting the model on the remaining data, and then predicting the left-out observation. The process is repeated for each observation in the dataset.

The average negative log-likelihood for LOO cross-validation is calculated as:
$$
E(\sigma) = -\frac{1}{N} \sum_{i=1}^{N} \log p_\sigma(x_i)
$$
where $p_\sigma(x_i)$ is the estimated density at $x_i$ when $x_i$ is left out of the density estimation.

### Steps to Solve:
1. **Calculate KDE for each observation**: For each $x_i$ in the dataset, estimate the density $p_\sigma(x_i)$ using the remaining $N-1$ observations.
2. **Compute Negative Log-Likelihood**: Compute the negative log-likelihood for each observation and then average these values.

### Example Calculation:
We will go through the steps to calculate the KDE and negative log-likelihood for different values of $\sigma$.

1. **Dataset**: $\{3.918, -6.35, -2.677, -3.003\}$
2. **KDE Formula**:
   $$
   p_\sigma(x) = \frac{1}{(N-1) \sqrt{2\pi\sigma^2}} \sum_{j \neq i} \exp\left( -\frac{(x - x_j)^2}{2\sigma^2} \right)
   $$

## Script

```python
import numpy as np

# Dataset
data = np.array([3.918, -6.35, -2.677, -3.003])

# Function to calculate KDE
def kde_leave_one_out(x_i, data, sigma):
    N = len(data)
    kde_sum = 0
    for x_j in data:
        if x_j != x_i:
            kde_sum += np.exp(-((x_i - x_j)**2) / (2 * sigma**2))
    return kde_sum / ((N - 1) * np.sqrt(2 * np.pi * sigma**2))

# Function to calculate E(sigma)
def calculate_E_sigma(data, sigma):
    N = len(data)
    negative_log_likelihoods = []
    for x_i in data:
        p_sigma = kde_leave_one_out(x_i, data, sigma)
        negative_log_likelihoods.append(-np.log(p_sigma))
    return np.mean(negative_log_likelihoods)

# Sigma values (example)
sigma_values = [0.1, 0.5, 1, 2, 3]

# Calculate E(sigma) for different sigma values
E_sigma_values = [calculate_E_sigma(data, sigma) for sigma in sigma_values]

# Output results
for sigma, E_sigma in zip(sigma_values, E_sigma_values):
    print(f"E({sigma}) = {E_sigma}")
```

### Choose plot
Result
```
E(0.1) = inf
E(0.5) = 28.776301235830914
E(1) = 8.784582878961187
E(2) = 4.072471660359569
E(3) = 3.3400933289379675
```
so when $\sigma = 2$ $E(2) = 4.07$
find the plot that matchs that point (2,4), it is plot 1.

---
## S23, Q27) Determine which value of λ in Figure 15 results in  $L = [−2.3 −2.3 −13.91]$.

![[Pasted image 20240521194243.png]]
![[Pasted image 20240521194246.png]]


### Breakdown

#### Problem Statement

A small 1-dimensional dataset with $N = 3$ observations has been subsampled from the Sound Classification dataset:

$$
X = \begin{bmatrix} -0.82 \\ 0.0 \\ 2.5 \end{bmatrix}
$$

A kernel density estimator (KDE) has been fitted to the small dataset with different $\lambda$ using the standard Gaussian kernel. We need to determine which value of $\lambda$ results in the test log likelihood $L$ values:

$$
L = \begin{bmatrix} -2.3 & -2.3 & -13.91 \end{bmatrix}
$$

The possible $\lambda$ values are (seen on the graphs):
- $\lambda = 1.15$
- $\lambda = 0.15$
- $\lambda = 0.21$
- $\lambda = 0.49$

#### Solution Explanation

The correct answer is $\lambda = 0.49$. To determine this, we perform a leave-one-out (LOO) procedure for computing the generalization error for different $\lambda$ values.

### Detailed Steps

1. **Leave-One-Out (LOO) Procedure**:
   - For each observation $x_i$, remove $x_i$ from the dataset and estimate the density at $x_i$ using the remaining $N-1$ observations.
   - The test log likelihood for the whole dataset is defined as:

   $$
   L(\lambda) = \frac{1}{N} \sum_{i=1}^{N} \log \left( \sum_{j \neq i} \frac{1}{N-1} \cdot \mathcal{N}(x_i | x_j, \lambda^2) \right)
   $$

2. **Kernel Density Estimation (KDE)**:
   - The KDE is calculated for different values of $\lambda$, and the log likelihood is evaluated for each $x_i$.

3. **Log Likelihood Calculation**:
   - For each $\lambda$, the individual log likelihoods $L_i$ are computed for each observation.
   - The values of $\lambda$ that best fit the log likelihood values given are determined.

4. **Choosing the Correct $\lambda$**:
   - Compare the computed log likelihood values for each $\lambda$ with the given log likelihoods $[-2.3, -2.3, -13.91]$.

### Example Calculation for $\lambda = 0.49$

Given the dataset $X = \begin{bmatrix} -0.82 \\ 0.0 \\ 2.5 \end{bmatrix}$, we perform the LOO procedure and calculate the log likelihoods for $\lambda = 0.49$.

- **Step 1**: Remove $x_1 = -0.82$
  - KDE using $x_2 = 0.0$ and $x_3 = 2.5$
  - Calculate $p(x_1)$ and the log likelihood.

- **Step 2**: Remove $x_2 = 0.0$
  - KDE using $x_1 = -0.82$ and $x_3 = 2.5$
  - Calculate $p(x_2)$ and the log likelihood.

- **Step 3**: Remove $x_3 = 2.5$
  - KDE using $x_1 = -0.82$ and $x_2 = 0.0$
  - Calculate $p(x_3)$ and the log likelihood.

By evaluating the above steps, the KDE with $\lambda = 0.49$ results in the log likelihood values $[-2.3, -2.3, -13.91]$, matching the given values.

### Conclusion

The value of $\lambda$ that results in the log likelihood values $[-2.3, -2.3, -13.91]$ is $\lambda = 0.49$.

#### Answer

$$
\boxed{\text{D. } \lambda = 0.49}
$$

This approach ensures a thorough understanding of the leave-one-out procedure and kernel density estimation, providing accurate determination of the correct $\lambda$ value.


```python
import numpy as np  
from scipy.stats import norm  
  
# Dataset  
X = np.array([-0.82, 0.0, 2.5])  
N = len(X)  
  
# Given log likelihood values  
given_log_likelihoods = np.array([-2.3, -2.3, -13.91])  
  
# KDE with different lambda values  
lambdas = [1.15, 0.15, 0.21, 0.49]  
log_likelihoods = {lam: [] for lam in lambdas}  
  
# Leave-One-Out procedure  
for lam in lambdas:  
    loo_log_likelihood = []  
    print(f"\nCalculating for lambda = {lam}")  
    for i in range(N):  
        # Leave one out  
        X_train = np.delete(X, i)  
        x_i = X[i]  
  
        # KDE estimation for x_i  
        p_xi_values = [norm.pdf(x_i, loc=x_j, scale=lam) for x_j in X_train]  
        p_xi = np.mean(p_xi_values)  
        loo_log_likelihood.append(np.log(p_xi))  
  
        # Debugging output  
        print(f"Left out {x_i}: KDE values = {p_xi_values}, p(x_i) = {p_xi}, log(p(x_i)) = {np.log(p_xi)}")  
  
    log_likelihoods[lam] = loo_log_likelihood  
  
print("\nLog Likelihoods for different lambdas:")  
for lam, ll in log_likelihoods.items():  
    print(f"lambda = {lam}: {ll}")  
  
  
# Compare each lambda's log likelihoods with the given log likelihoods  
def total_difference(log_likelihoods, given_log_likelihoods):  
    return np.sum(np.abs(np.array(log_likelihoods) - given_log_likelihoods))  
  
  
best_lambda = min(lambdas, key=lambda lam: total_difference(log_likelihoods[lam], given_log_likelihoods))  
  
print(f"\nThe best lambda is {best_lambda}")
```

