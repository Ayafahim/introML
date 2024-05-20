
## F23, Q21)





### Question 21
Consider the synthetic 1D dataset with four training observations:

\[ X = \{-3, -1, 5, 6\} \]

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
