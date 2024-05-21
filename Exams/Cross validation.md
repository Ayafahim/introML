
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

---
## S23, Q6) Determine the error rates on the test set in each of the 2 outer folds based on the information in Table 3, Table 4 and Table 2

![[Pasted image 20240520201224.png]]


## Solution no script


We consider the generalization error of K-nearest neighbour (KNN) classifiers with $K = [1, 3, 4]$ neighbors based on the observations in Table 2. We apply two-layer cross-validation to estimate the generalization error with 2 outer and 5 inner folds. Tables 3 and 4 provide partial results of the associated two-layer cross-validation. It is noted that, with other things being equal, a model with the lowest value of $K$ is preferred in this context. The specific split of the dataset in the two outer folds is also known:

- Outer fold $i = 1$ training set: $o_1, o_2, o_4, o_8, o_9$.
- Outer fold $i = 2$ training set: $o_3, o_5, o_6, o_7, o_{10}$.

Determine the error rates on the test set in each of the 2 outer folds based on the information in Table 3, Table 4, and Table 2.

### Solution Steps:

1. **Tables Information:**
   - Table 2: Pairwise Euclidean distances between the observations.
   - Table 3: Error rates in inner folds for outer fold 1.
   - Table 4: Error rates in inner folds for outer fold 2.

2. **Optimal $K$ Selection:**
   - For outer fold 1, the error rates in the inner folds for $K = 1, 3, 4$ are given in Table 3.
   - For outer fold 2, the error rates in the inner folds for $K = 1, 3, 4$ are given in Table 4.
   - The optimal $K$ is chosen based on the minimum error rate across the inner folds.

3. **Calculating Test Error Rates:**
   - For each outer fold, we use the optimal $K$ found from the inner folds to calculate the test error rate on the test set of the respective outer fold.

#### Outer Fold 1:
- Training set: $o_1, o_2, o_4, o_8, o_9$
- Test set: $o_3, o_5, o_6, o_7, o_{10}$

From Table 3, we observe the inner fold error rates for different values of $K$:
- $K = 1$: $[0, 0, 1, 0, 0]$
- $K = 3$: $[1, 1, 1, 1, 1]$
- $K = 4$: $[1, 1, 1, 0, 0]$

Optimal $K$ is the one with the minimum error rate. Here, $K = 1$ has the minimum error rate $\frac{0+0+1+0+0}{5} = 0.2$.

#### Outer Fold 2:
- Training set: $o_3, o_5, o_6, o_7, o_{10}$
- Test set: $o_1, o_2, o_4, o_8, o_9$

From Table 4, we observe the inner fold error rates for different values of $K$:
- $K = 1$: $[0, 0, 1, 0, 1]$
- $K = 3$: $[0, 0, 1, 0, 0]$
- $K = 4$: $[0, 0, 1, 0, 0]$

Optimal $K$ is the one with the minimum error rate. Here, $K = 3$ has the minimum error rate $\frac{0+0+1+0+0}{5} = 0.2$.

Using the selected $K$ values, the error rates on the test set for each outer fold are computed.

#### Test Error Calculation:
- For outer fold 1 with $K = 1$:
  - Use the KNN classifier with $K = 1$ on the test set observations $o_3, o_5, o_6, o_7, o_{10}$.
  - The error rate calculation involves classifying these test observations and computing the misclassification rate.

- For outer fold 2 with $K = 3$:
  - Use the KNN classifier with $K = 3$ on the test set observations $o_1, o_2, o_4, o_8, o_9$.
  - Similarly, calculate the misclassification rate.

### Conclusion:
From the exam solution, the correct answer is:

- $E_{\text{test}, i=1} = 0.2$
- $E_{\text{test}, i=2} = 0.6$

Thus, the correct option is:

**D. $E_{\text{test}, i=1} = 0.2$ and $E_{\text{test}, i=2} = 0.6$**

### Solution with script

- **Outer Fold 1:**
    
    - Training set: {o1, o2, o4, o8, o9} which corresponds to indices `[0, 1, 3, 7, 8]`
    - Test set: {o3, o5, o6, o7, o10} which corresponds to indices `[2, 4, 5, 6, 9]`
- **Outer Fold 2:**
    
    - Training set: {o3, o5, o6, o7, o10} which corresponds to indices `[2, 4, 5, 6, 9]`
    - Test set: {o1, o2, o4, o8, o9} which corresponds to indices `[0, 1, 3, 7, 8]`


```python 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Distance matrix from the exam (Table 2)
distance_matrix = np.array([
    [0.0, 1.3, 4.1, 3.8, 4.5, 2.4, 3.2, 2.7, 3.0, 3.9],
    [1.3, 0.0, 3.2, 3.1, 4.7, 2.3, 2.6, 2.2, 2.7, 4.2],
    [4.1, 3.2, 0.0, 0.4, 4.9, 2.7, 1.1, 1.6, 2.4, 4.8],
    [3.8, 3.1, 0.4, 0.0, 4.6, 2.5, 0.9, 1.3, 2.1, 4.5],
    [4.5, 4.7, 4.9, 4.6, 0.0, 3.1, 4.4, 3.7, 2.8, 2.3],
    [2.4, 2.3, 2.7, 2.5, 3.1, 0.0, 1.8, 1.2, 0.9, 2.8],
    [3.2, 2.6, 1.1, 0.9, 4.4, 1.8, 0.0, 1.0, 1.7, 4.1],
    [2.7, 2.2, 1.6, 1.3, 3.7, 1.2, 1.0, 0.0, 1.1, 3.6],
    [3.0, 2.7, 2.4, 2.1, 2.8, 0.9, 1.7, 1.1, 0.0, 2.9],
    [3.9, 4.2, 4.8, 4.5, 2.3, 2.8, 4.1, 3.6, 2.9, 0.0]
])

# Class labels from the exam
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# K values to evaluate
k_values = [1, 3, 4]

# Outer cross-validation split
# These values are derived from the problem statement 
# The indexes of o1,o2....

outer_train_1 = [0, 1, 3, 7, 8]
outer_test_1 = [2, 4, 5, 6, 9]
outer_train_2 = [2, 4, 5, 6, 9]
outer_test_2 = [0, 1, 3, 7, 8]

# Inner cross-validation setup
kf = KFold(n_splits=5)

# Function to get indices of K nearest neighbors
def get_knn_indices(dist_matrix, k):
    return np.argsort(dist_matrix, axis=1)[:, 1:k+1]

# Evaluate KNN classifier
def evaluate_knn(train_indices, test_indices, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    knn.fit(distance_matrix[train_indices][:, train_indices], labels[train_indices])
    preds = knn.predict(distance_matrix[test_indices][:, train_indices])
    return accuracy_score(labels[test_indices], preds)

# Two-layer cross-validation
def two_layer_cv(train_indices, k_values):
    best_k = None
    best_score = -1
    for k in k_values:
        scores = []
        for train_inner, test_inner in kf.split(train_indices):
            train_inner_indices = [train_indices[i] for i in train_inner]
            test_inner_indices = [train_indices[i] for i in test_inner]
            score = evaluate_knn(train_inner_indices, test_inner_indices, k)
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    return best_k

# Outer fold 1
best_k_1 = two_layer_cv(outer_train_1, k_values)
error_rate_1 = 1 - evaluate_knn(outer_train_1, outer_test_1, best_k_1)

# Outer fold 2
best_k_2 = two_layer_cv(outer_train_2, k_values)
error_rate_2 = 1 - evaluate_knn(outer_train_2, outer_test_2, best_k_2)

# Results
print(f"Error rate for outer fold 1: {error_rate_1}")
print(f"Error rate for outer fold 2: {error_rate_2}")
```