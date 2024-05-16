
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