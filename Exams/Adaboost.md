
## S19, Q24) Given this information, how will the AdaBoost update the weights w?

![[Pasted image 20240516030947.png]]
Look at the table and determine correctly classified vs misclassified

```python
import numpy as np  
  
# Initial weights  
N = 7  
weights = np.full(N, 1/N)  
  
# Mock results from a classifier (True = correctly classified, False = misclassified)  
results = np.array([True, False, False, False, True, False, False])  
  
# Calculate error rate  
epsilon = np.sum(weights[~results])  
  
# Calculate classifier performance  
alpha = 0.5 * np.log((1 - epsilon) / epsilon)  
  
# Update weights  
weights[results] *= np.exp(-alpha)  
weights[~results] *= np.exp(alpha)  
  
# Normalize weights  
weights /= np.sum(weights)  
  
print("Updated Weights:", weights)
```



---
## F16, Q24) What will the updated weights be for these misclassified observations according to the AdaBoost algorithm?
![[Pasted image 20240521153409.png]]
![[Pasted image 20240521153406.png]]

```python
import numpy as np # type: ignore  
  
def adaboost_weight_update(N, misclassified_count, initial_weight):  
    # Calculate the error rate (epsilon)  
    epsilon_t = misclassified_count / N  
  
    # Calculate alpha_t using the formula: 0.5 * log((1 - epsilon) / epsilon)  
    alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)  
  
    # Calculate the new weights for misclassified observations  
    new_weight_misclassified = initial_weight * np.exp(alpha_t)  
  
    # Calculate the new weights for correctly classified observations  
    new_weight_correct = initial_weight * np.exp(-alpha_t)  
  
    # Calculate normalization factor Z_t to ensure weights sum to 1  
    Z_t = (N - misclassified_count) * new_weight_correct + misclassified_count * new_weight_misclassified  
  
    # Normalize the weights  
    normalized_weight_misclassified = new_weight_misclassified / Z_t  
    normalized_weight_correct = new_weight_correct / Z_t  
  
    return normalized_weight_misclassified, normalized_weight_correct, Z_t  
  
# Parameters  
N = 25  # Total number of observations  
misclassified_count = 5  # Number of misclassified observations  
initial_weight = 1 / N  # Initial uniform weight  
  
# Get the updated weights for misclassified observations  
updated_weight_misclassified, updated_weight_correct, normalization_factor = adaboost_weight_update(N, misclassified_count, initial_weight)  
print(f"Normalization factor: {normalization_factor:.3f}")  
print(f"The updated weights for classifed observations based on Adaboost is: {updated_weight_correct:.3f}")  
print(f"The updated weights for misclassifed observations based on Adaboost is: {updated_weight_misclassified:.3f}")
```