

## S19, Q23) According to the GMM, what is the probability an observation at x0 = 3.19 is assigned to cluster k = 2?

![[Pasted image 20240516031722.png]]

To solve Question 23, we need to understand how to calculate the probability that an observation at \( x_0 = 3.19 \) is assigned to cluster \( k = 2 \) in a Gaussian Mixture Model (GMM). Here’s a step-by-step breakdown of how to approach this problem:

### Given:
- Mixture weights: $( w_1 = 0.19 ), ( w_2 = 0.34 ), ( w_3 = 0.48 )$
- Means:$( \mu_1 = 3.177 ), ( \mu_2 = 3.181 ), ( \mu_3 = 3.184 )$
- Variances: $( \sigma_1^2 = 0.0062 ), ( \sigma_2^2 = 0.0076 ), ( \sigma_3^2 = 0.0075 )$
- Observation: $( x_0 = 3.19 )$

We need to calculate the probability that \( x_0 = 3.19 \) is assigned to cluster \( k = 2 \).

### Steps to Calculate the Probability

1. **Calculate the Gaussian Probability Density Function (PDF)**:
   - The probability density for a Gaussian distribution is given by:
     $$
     N(x \mid \mu_k, \sigma_k^2) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left( -\frac{(x - \mu_k)^2}{2\sigma_k^2} \right)
     $$

2. **Calculate the PDF for Each Cluster**:
   - For \( k = 1 \):
     $$
     N(3.19 \mid 3.177, 0.0062) = \frac{1}{\sqrt{2\pi \cdot 0.0062}} \exp\left( -\frac{(3.19 - 3.177)^2}{2 \cdot 0.0062} \right)
     $$
   - For \( k = 2 \):
     $$
     N(3.19 \mid 3.181, 0.0076) = \frac{1}{\sqrt{2\pi \cdot 0.0076}} \exp\left( -\frac{(3.19 - 3.181)^2}{2 \cdot 0.0076} \right)
     $$
   - For \( k = 3 \):
     $$
     N(3.19 \mid 3.184, 0.0075) = \frac{1}{\sqrt{2\pi \cdot 0.0075}} \exp\left( -\frac{(3.19 - 3.184)^2}{2 \cdot 0.0075} \right)
     $$

3. **Calculate the Posterior Probability**:
   - The posterior probability that \( x_0 \) belongs to cluster \( k \) is given by:
     $$
     P(k \mid x_0) = \frac{w_k N(x_0 \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^{3} w_j N(x_0 \mid \mu_j, \sigma_j^2)}
     $$
![[Pasted image 20240516031822.png]]

Note i might be like a decimal off:

```python
import numpy as np  
  
# Given parameters  
w = [0.19, 0.34, 0.48]  
mu = [3.177, 3.181, 3.184]  
sigma2 = [0.0062, 0.0076, 0.0075]  
x0 = 3.19  
  
# Calculate Gaussian PDF for each cluster  
def gaussian_pdf(x, mu, sigma2):  
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(- (x - mu)**2 / (2 * sigma2))  
  
# Calculate the PDFs  
pdfs = [gaussian_pdf(x0, mu[i], sigma2[i]) for i in range(3)]  
  
# Calculate the posterior probability for each cluster  
posterior_probs = [w[i] * pdfs[i] for i in range(3)]  
posterior_sum = sum(posterior_probs)  
posterior_probs = [prob / posterior_sum for prob in posterior_probs]  
  
# Probability that x0 belongs to cluster 2 (index 1 since it's zero-indexed)  
prob_cluster_2 = posterior_probs[1]  
  
print(f"Probability that x0 = {x0} belongs to cluster 2: {prob_cluster_2:.3f}")
```

---

## F17, Q12) What is the probability that observation O8 is assigned to cluster 2 according to the GMM?
Similar to question above 
![[Pasted image 20240521155222.png]]




> [!NOTE]
> NOTE that values are not the correct ones for this question.


```python
import numpy as np # type: ignore

# Parameters for the Gaussian Mixture Model
#TODO update weights means variances based on task
weights = np.array([0.15, 0.53, 0.32])  # Weights for each cluster
means = np.array([1.45, 1.6, 1.25])    # Means for each cluster, top of the graph y-axis for plots
variances = np.array([1.3, 1.2, 1.4])  # Variances for each cluster, THESE CAN BE READ ON THE GRAPH --> ESTIMATE THE 95% QUANTILE

# Function to compute Gaussian density
def gaussian_density(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

#Observation for O8 TODO: update x based on the observation
x = 0

# Compute the density of this observation for each cluster
densities = np.array([gaussian_density(x, mu, sigma2) for mu, sigma2 in zip(means, variances)])

# Calculate the weighted densities to find the contribution from each cluster
weighted_densities = weights * densities

# Compute the probability of belonging to cluster 2
# if another cluster is needed, update the index e.g weighted_densities[2] for cluster 3
probability_cluster_2 = weighted_densities[1] / sum(weighted_densities)

print("Probability of O8 belonging to cluster 2:", probability_cluster_2)
```