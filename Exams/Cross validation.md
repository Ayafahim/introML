
## S19, Q22) How many models were trained to compose the table?
![[Pasted image 20240516031236.png]]
![[Pasted image 20240516031300.png]]
### Key Components of the Solution:

1. **Outer Fold (\(K1\))**: Number of outer folds.
2. **Inner Fold (\(K2\))**: Number of inner folds for cross-validation.
3. **Hyperparameter Values (\(S\))**: Number of hyperparameter values to test.

### Given:
- **\(K1 = 4\)**: 4 outer folds.
- **\(K2 = 5\)**: 5 inner folds for cross-validation.
- **\(S = 5\)**: 5 values to test for each hyperparameter (either $(n_h)$ for the neural network or $(\lambda)$ for the logistic regression).

### Two-level Cross-validation Calculation:
1. **Inner Cross-validation Models**:
   - For each outer fold, \(K2\) inner folds are used to evaluate each of the \(S\) hyperparameter values.
   - Therefore, for each outer fold, the number of models trained is $(K2 \times S)$.

2. **Outer Fold Models**:
   - For each outer fold, after determining the optimal hyperparameter value, an additional model is trained on the entire outer training set and tested on the outer test set.

3. **Total Models for Each Outer Fold**:
   - $(K2 \times S)$ models for inner cross-validation + 1 final model for testing = $(K2 \times S + 1).$

### Calculation:
1. **Number of Models per Outer Fold**:
   - $(K2 \times S + 1 = 5 \times 5 + 1 = 26$).

2. **Total Models for All Outer Folds**:
   - $(K1 \times (K2 \times S + 1) = 4 \times 26 = 104).$

3. **Total for Both Models**:
   - This calculation is done separately for both the neural network and the logistic regression model.
   - Therefore, the total number of models for both models combined is \(2 \times 104 = 208\).

### Conclusion:
- The total number of models trained for each method (neural network or logistic regression) is \(104\).
- Since the problem asks for the total number of models trained for both methods, the correct number is \(208\).

### Correct Answer:
\[ \boxed{208 \text{ models}} \]
Therefore, the correct choice is **A. 208 models**.

### Summary of Solution:

1. **Outer Fold**: 4 splits.
2. **Inner Fold**: 5 splits.
3. **Hyperparameter Values**: 5 values for each method.
4. **Models per Outer Fold**: \(5 \times 5 + 1 = 26\).
5. **Total Models for One Method**: \(4 \times 26 = 104\).
6. **Total Models for Both Methods**: \(2 \times 104 = 208\).

---

## F23, Q9) Which one the following options is the correct estimate of the generalization error?

![[Pasted image 20240518003710.png]]

```python
import numpy as np # type: ignore  
  
# Data for outer fold i=1  
inner_folds_i1 = {  
    "Model 1": [0.12, 0.21, 0.22, 0.23, 0.15],  
    "Model 2": [0.30, 0.11, 0.15, 0.30, 0.28],  
    "Model 3": [0.21, 0.14, 0.26, 0.17, 0.26]  
}  
test_errors_i1 = {  
    "Model 1": 0.24,  
    "Model 2": 0.17,  
    "Model 3": 0.22  
}  
  
# Data for outer fold i=2  
inner_folds_i2 = {  
    "Model 1": [0.28, 0.18, 0.19, 0.27, 0.12],  
    "Model 2": [0.16, 0.20, 0.27, 0.30, 0.25],  
    "Model 3": [0.13, 0.16, 0.21, 0.17, 0.13]  
}  
test_errors_i2 = {  
    "Model 1": 0.19,  
    "Model 2": 0.16,  
    "Model 3": 0.25  
}  
  
# Calculate average validation error for each model in each outer fold  
def average_validation_error(inner_folds):  
    return {model: np.mean(errors) for model, errors in inner_folds.items()}  
  
avg_val_error_i1 = average_validation_error(inner_folds_i1)  
avg_val_error_i2 = average_validation_error(inner_folds_i2)  
  
# Identify the best model in each outer fold  
best_model_i1 = min(avg_val_error_i1, key=avg_val_error_i1.get)  
best_model_i2 = min(avg_val_error_i2, key=avg_val_error_i2.get)  
  
# Retrieve the test errors for the best models  
test_error_best_model_i1 = test_errors_i1[best_model_i1]  
test_error_best_model_i2 = test_errors_i2[best_model_i2]  
  
# Calculate the generalization error  
generalization_error = np.mean([test_error_best_model_i1, test_error_best_model_i2])  
  
# Output the result  
print(f"Generalization Error (E_gen): {generalization_error:.3f}")
```
