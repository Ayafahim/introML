
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

---
## S20, 27)



We need to use AdaBoost to classify two test observations $\mathbf{x}_{\text{test1}}$ and $\mathbf{x}_{\text{test2}}$ using a 1-NN classifier over four rounds of boosting. The question provides the predictions of the AdaBoost classifiers and the weighted error rates $\epsilon_t$ for each round.

Here's a step-by-step guide to solve the problem:

1. **Understanding AdaBoost**:
   - AdaBoost combines weak classifiers to form a strong classifier.
   - For each round $t$, the weak classifier’s prediction and its weighted error rate $\epsilon_t$ are provided.
   - The final prediction is a weighted majority vote of the weak classifiers’ predictions.

2. **Calculation Details**:
   - The weight for each classifier is calculated using:
     $$
     \alpha_t = \frac{1}{2} \log\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
     $$
   - The final prediction is made by summing the weighted predictions and applying the sign function.

### Steps to Solve the Problem:
1. **Extract Predictions and Error Rates**:
   - From Table 6 in the exam, extract the predictions for each test observation for the 4 rounds and the corresponding error rates.

2. **Calculate Classifier Weights**:
   - Compute $\alpha_t$ for each round using the given error rates.

3. **Aggregate Weighted Predictions**:
   - Sum the weighted predictions for each test observation.
   - Apply the sign function to determine the final class.

### Example Calculation

Given:
- Predictions for $\mathbf{x}_{\text{test1}}$: $0, 1, 0, 0$
- Predictions for $\mathbf{x}_{\text{test2}}$: $0, 1, 1, 1$
- Error rates $\epsilon_t$: $0.417, 0.243, 0.307, 0.534$

```python
import numpy as np  
  
  
def adaboost_classification(predictions, error_rates):  
    # Calculate classifier weights  
    alphas = 0.5 * np.log((1 - np.array(error_rates)) / np.array(error_rates))  
  
    # Compute weighted sum of predictions for each test observation  
    weighted_sums = np.dot(predictions, alphas)  
  
    # Final prediction is the sign of the weighted sum  
    final_predictions = np.sign(weighted_sums)  
  
    # Convert from -1, 1 to 0, 1  
    final_predictions = (final_predictions + 1) // 2  
    return final_predictions  
  
  
# Predictions for each test observation  
predictions_test1 = [0, 1, 0, 0]  
predictions_test2 = [0, 1, 1, 1]  
  
# Convert 0/1 predictions to -1/1 for consistency with AdaBoost  
predictions_test1 = np.array(predictions_test1) * 2 - 1  
predictions_test2 = np.array(predictions_test2) * 2 - 1  
  
# Error rates for each round  
error_rates = [0.417, 0.243, 0.307, 0.534]  
  
# Calculate final predictions  
final_pred_test1 = adaboost_classification(predictions_test1, error_rates)  
final_pred_test2 = adaboost_classification(predictions_test2, error_rates)  
  
print(f"Final prediction for x_test1: {final_pred_test1}")  
print(f"Final prediction for x_test2: {final_pred_test2}")
```
