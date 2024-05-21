
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
## S23, Q25) What is the importance of the classifier f1?

![[Pasted image 20240521031659.png]]
## Script

```python
import numpy as np  
  
# Binarized dataset (Table 8)  
data = np.array([  
    [1, 1, 0, 1, 0, 0, 'C1'],  
    [1, 0, 1, 1, 0, 0, 'C1'],  
    [0, 1, 1, 1, 1, 1, 'C1'],  
    [0, 1, 1, 1, 1, 1, 'C1'],  
    [1, 1, 0, 1, 1, 0, 'C2'],  
    [0, 1, 1, 0, 1, 1, 'C2'],  
    [0, 1, 1, 1, 0, 1, 'C2'],  
    [0, 1, 0, 1, 1, 1, 'C2'],  
    [0, 1, 0, 1, 1, 1, 'C2'],  
    [1, 1, 1, 1, 1, 0, 'C2']  
])  
  
# Classifier f1: f1(b1, b2, b3, b4, b5, b6) = C1 if b3 = 1 and b4 = 1, otherwise C2  
#TODO update based on what the task says  
def classifier_f1(b):  
    b3, b4 = b[2], b[3]  
    if b3 == 1 and b4 == 1:  
        return 'C1'  
    else:  
        return 'C2'  
  
# Calculate predictions and actual classes  
predictions = []  
actual_classes = data[:, -1]  
for i in range(data.shape[0]):  
    b = data[i, :-1].astype(int)  
    prediction = classifier_f1(b)  
    predictions.append(prediction)  
  
# Calculate error rate epsilon_1  
misclassified = np.sum(predictions != actual_classes)  
total = data.shape[0]  
epsilon_1 = misclassified / total  
  
# Calculate the importance alpha_1 of the classifier f1  
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)  
  
# Print the results  
print(f"Predictions: {predictions}")  
print(f"Actual Classes: {actual_classes.tolist()}")  
print(f"Misclassified: {misclassified}")  
print(f"Total Observations: {total}")  
print(f"Error Rate (epsilon_1): {epsilon_1:.2f}")  
print(f"Importance (alpha_1): {alpha_1:.2f}")
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