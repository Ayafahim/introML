## F23, Q12) standardize the input before learning the optimal bias  $w^∗_0$ and parameter $w^*_1$ for a regularized linear regression model with λ = 0.7, where we consider the formulation from Section 14.1. What is the prediction of the trained model for the input $x_2$?

### Solution

### Given Information
- Input values: $([-3.4, -1.3, 0.5, 2.4, 4.2])$
- Output values: $([-2.9, -0.4, 0.7, 2.5, 4.5])$
- Regularization parameter: $(\lambda = 0.7)$
- We are standardizing the input before fitting the model.

### Steps in the Solution

1. **Standardize the Input:**
   - Compute the mean $(\mu)$ and standard deviation $(\sigma)$ of $(x)$:

     $$\mu = \frac{-3.4 + (-1.3) + 0.5 + 2.4 + 4.2}{5} = 0.48$$

     $$\sigma \approx 2.9895$$

   - Standardize \( x \):

     $$\tilde{x}_i = \frac{x_i - \mu}{\sigma}$$

2. **Formulate the Regularized Linear Regression:**
   - Use the formula for finding the optimal parameters:

     $$w_0^* = \frac{1}{5} \sum y_i = \frac{-2.9 + (-0.4) + 0.7 + 2.5 + 4.5}{5} = 0.88$$

     $$w_1^* = \frac{\sum \tilde{x}_i (y_i - w_0^*)}{\sum \tilde{x}_i^2 + \lambda}$$

3. **Calculate $(\tilde{x}_i)$ values:**
   - Standardized values:

     $$\tilde{x}_1 = \frac{-3.4 - 0.48}{2.9895} \approx -1.302$$

     $$\tilde{x}_2 = \frac{-1.3 - 0.48}{2.9895} \approx -0.595$$

     $$\tilde{x}_3 = \frac{0.5 - 0.48}{2.9895} \approx 0.007$$

     $$\tilde{x}_4 = \frac{2.4 - 0.48}{2.9895} \approx 0.643$$

     $$\tilde{x}_5 = \frac{4.2 - 0.48}{2.9895} \approx 1.244$$

4. **Calculate $( w_1^* )$:**
   - Sum up the numerator and denominator separately:

     $$\text{Numerator} = \sum \tilde{x}_i (y_i - w_0^*)$$
     $$= (-1.302 \cdot (-2.9 - 0.88)) + (-0.595 \cdot (-0.4 - 0.88)) + (0.007 \cdot (0.7 - 0.88)) + (0.643 \cdot (2.5 - 0.88)) + (1.244 \cdot (4.5 - 0.88))$$
     $$= (1.302 \cdot 3.78) + (0.595 \cdot 1.28) + (0.007 \cdot -0.18) + (0.643 \cdot 1.62) + (1.244 \cdot 3.62)$$
     $$= 4.92 + 0.76 - 0.001 + 1.04 + 4.5 \approx 11.22$$

     $$\text{Denominator} = \sum \tilde{x}_i^2 + \lambda$$
     $$= (-1.302)^2 + (-0.595)^2 + (0.007)^2 + (0.643)^2 + (1.244)^2 + 0.7$$
     $$= 1.695 + 0.354 + 0.00005 + 0.413 + 1.548 + 0.7 \approx 4.71$$

   - Calculate $( w_1^* )$:

     $$w_1^* = \frac{11.22}{4.71} \approx 2.38$$

5. **Prediction for $( x_2$):**
   - Standardized value of $( x_2)$:

     $$\tilde{x}_2 = -0.595$$

   - Prediction using the trained model:

     $$\hat{y}_2 = w_0^* + w_1^* \cdot \tilde{x}_2 = 0.88 + 2.38 \cdot (-0.595) = 0.88 - 1.415 = -0.535$$

### Conclusion
The correct answer is:
B. $( \hat{y} \approx -0.54)$

### Summary Notes:
1. **Standardization:**
   - Subtract the mean and divide by the standard deviation.

2. **Regularized Linear Regression:**
   - Use the formula $( w_0^* = \frac{1}{5} \sum y_i )$
   - Use $( w_1^* = \frac{\sum \tilde{x}_i (y_i - w_0^*)}{\sum \tilde{x}_i^2 + \lambda})$

### Script
```python 
import numpy as np  
  
  
# Synthetic dataset  
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])  
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])  
  
# Standardization parameters  
mu = np.mean(x)  
sigma = np.std(x, ddof=1)  # Using ddof=1 for sample standard deviation  
  
# Standardize x  
x_standardized = (x - mu) / sigma  
  
# Lambda for regularization  
lambda_ = 0.7  
  
# Calculate the intercept w0 as per the provided formula  
w0 = np.mean(y)  # Since the formula was w0 = 1/5 * sum(yi), which simplifies to the mean of y  
  
# Calculate w1 using the corrected formula  
w1 = np.sum(x_standardized * (y - w0)) / (np.sum(x_standardized**2) + lambda_)  
  
# Predicting for x2 (value at index 1 in the original x array, corresponding to -1.3)  
x2_standardized = (-1.3 - mu) / sigma # This needs to be looked in the table to make sure it is the right number  
y_pred = w1 * x2_standardized + w0  
  
print("w0:", w0)  
print("w1:", w1)  
print("Predicted y for x2:", y_pred)
```

---

## S19,Q11) Suppose the resulting decision boundary is as shown in Figure 6, what are the weights?

![[Pasted image 20240521130727.png]]![[Pasted image 20240521130800.png]]

## Explained

### The Components

1. **Bias Vector (\(b\))**:
   - This is like an initial score or adjustment given to the data before we even start. In your problem, it's `[-0.0, -1.0]`, meaning the first 'bucket' doesn't get an initial adjustment, but the second one starts with a penalty of -1.

2. **Weights and Projections**:
   - Imagine you have a set of scales or measures to weigh the importance of different features of your data point. The weights tell you how to balance these features to get a score. You do this for each class.
   - **Projection** is a fancy word here for the score you get after applying these weights to your data's features. Higher scores suggest the data fits well into that class (or bucket).

3. **The Point and its Projection**:
   - The point here seems to be represented by `[-0.78, 0.29, 0.0]`, `[-0.9, -0.05, -0.0]`, etc., for different settings or models. These are scores for how well the data fits into each bucket, with each projection being a different method or model's guess.

4. **Choosing the Class**:
   - You pick the bucket with the highest score. Since the model gives a score for each bucket (class), the one with the highest score is deemed the most likely one for your data point. For example, if the scores are `[-0.78, 0.29, 0.0]`, the highest score here is `0.29`, so the data fits best in the second bucket according to this model.

6. **Final Prediction**:
   - You then look at each method's predictions. For instance, the first method might say the point belongs to Class 2 (since 0.29 is the highest score in the first set of projections). You do this for each method. Finally, you check which method predicted correctly by comparing with the actual class of the point (given in the figure or elsewhere in your materials). 
   - E.g in the next one 0.5 is the highest, which indicates the point belongs to class 1 which is wrong.

## Script

```python
import numpy as np  
  
# Define the input point b with 2 coordinates  
b = np.array([0.0, -1.0])  
  
# Define the weight matrices for each option  
weights = {  
    'A': {  
        'w1': np.array([-0.77, -5.54, 0.01]),  
        'w2': np.array([0.26, -2.09, -0.03])  
    },  
    'B': {  
        'w1': np.array([0.51, 1.65, 0.01]),  
        'w2': np.array([0.1, 3.8, 0.04])  
    },  
    'C': {  
        'w1': np.array([-0.9, -4.39, 0.0]),  
        'w2': np.array([-0.09, -2.45, -0.04])  
    },  
    'D': {  
        'w1': np.array([-1.22, -9.88, -0.01]),  
        'w2': np.array([-0.28, -2.9, -0.01])  
    }  
}  
  
  
# Function to compute the projections y1 and y2  
def compute_projections(b, w):  
    # Append a bias term of 1 to b to handle the third term in the weight vector  
    b_with_bias = np.append(b, 1)  
    y1 = np.dot(w['w1'], b_with_bias)  
    y2 = np.dot(w['w2'], b_with_bias)  
    return np.array([y1, y2])  
  
  
# Function to apply the softmax transformation  
def softmax(x):  
    exp_x = np.exp(x - np.max(x))  
    return exp_x / exp_x.sum()  
  
  
# Function to compute the class probabilities  
def compute_probabilities(projections):  
    y_hat = np.append(projections, 0)  # Adding y3=0 for the third class  
    return softmax(y_hat)  
  
  
# Expected results for verification  
expected_projections = {  
    'A': np.array([-0.78, 0.29, 0.0]),  
    'B': np.array([0.5, 0.06, 0.0]),  
    'C': np.array([-0.9, -0.05, -0.0]),  
    'D': np.array([-1.21, -0.27, -0.0])  
}  
  
# Compute and display results for each option  
for option, w in weights.items():  
    projections = compute_projections(b, w)  
    probabilities = compute_probabilities(projections)  
    predicted_class = np.argmax(probabilities) + 1  # Adding 1 to match class labels (1, 2, 3)  
  
    print(f"Option {option}:")  
    print(f"Projections: {projections}")  
    print(f"Probabilities: {probabilities}")  
    print(f"Predicted Class: {predicted_class}")  
    print(f"Expected Projections: {expected_projections[option]}")
```

---
## F16,Q13) Which one of the following solutions to w corresponds to the correct value of λ?

![[Pasted image 20240521151117.png]]
```python
import numpy as np # type: ignore  
  
# Define the weight vectors as numpy arrays  
w_a = np.array([0.0538, 0.0558, 0.1861, -0.0596])  
w_b = np.array([0.0089, 0.0931, 0.1093, 0.0417])  
w_c = np.array([0.2811, 0.0445, 0.3379, -0.4626])  
w_d = np.array([0.0167, 0.0698, 0.1354, 0.0403])  
  
# Compute the norms of each weight vector  
norms = {  
    'w_a': np.linalg.norm(w_a),  
    'w_b': np.linalg.norm(w_b),  
    'w_c': np.linalg.norm(w_c),  
    'w_d': np.linalg.norm(w_d)  
}  
  
# Print the norms for inspection  
for key, value in norms.items():  
    print(f"Norm of {key}: {value}")  
  
# Sort the dictionary by norms (values), get a list of keys based on sorted norms  
sorted_weights_by_norm = sorted(norms, key=norms.get, reverse=True)  # largest norm first if reverse is True  
  
# Map sorted weights to their corresponding lambda values (sorted by lambda: 1, 10, 100, 1000)  
lambda_mapping = {10: sorted_weights_by_norm[1], 100: sorted_weights_by_norm[2], 1000: sorted_weights_by_norm[3], 1: sorted_weights_by_norm[0]}  
  
# Output the corresponding weight for lambda = 10  
#TODO: Rememebr to up lambda_mapping value based on what lambda is  
print(f"The weight vector corresponding to lambda = 10 is: {lambda_mapping[10]}")
```
---
## S23, Q19) Which one of the following transformations was used  when learning w0 and w? #ridgeRegression

![[Pasted image 20240521173634.png]]
## Solution

The problem involves a ridge regression model to predict a perceived annoyance measure (PAM) using transformed observations. Here are the details:

Given:
- A dataset with $N = 4$ observations and a single attribute.
- Observations $\mathbf{x}$:
  $$
  \begin{bmatrix}
  -0.5 \\
  0.39 \\
  1.19 \\
  -1.08
  \end{bmatrix}
  $$
- Corresponding perceived annoyance measures $y_r$:
  $$
  \begin{bmatrix}
  -0.86 \\
  -0.61 \\
  1.37 \\
  0.10
  \end{bmatrix}
  $$
- Ridge regression parameters learned using a specific transformation of $x$, with $\lambda = 0.25$:
  - $E_{\lambda=0.25} \approx 0.2$
  - $w_0 = 0.0$
  - $\mathbf{w} = \begin{bmatrix} 0.39 & 0.77 \end{bmatrix}$

### Goal

Determine which transformation was used to obtain the results.

### Possible Transformations

A. $\mathbf{x}_i = \begin{bmatrix} x_i \\ x_i^3 \end{bmatrix}$  
B. $\mathbf{x}_i = \begin{bmatrix} x_i \\ \sin(x_i) \\ x_i^2 \end{bmatrix}$  
C. $\mathbf{x}_i = \begin{bmatrix} x_i \\ \sin(x_i) \end{bmatrix}$  
D. $\mathbf{x}_i = \begin{bmatrix} x_i \\ x_i^2 \end{bmatrix}$  
E. Don’t know

### Approach

1. **Form the design matrix**: For each transformation, create the design matrix $\mathbf{X}$ using the observations $\mathbf{x}$.
2. **Standardize the design matrix**: Subtract the mean and divide by the standard deviation for each column.
3. **Calculate the ridge regression cost**: Compute the total cost $E_\lambda$ using the formula:
$$
E_\lambda = \sum_{i=1}^{N} (y_{r,i} - \mathbf{x}_i^T \mathbf{w} - w_0)^2 + \lambda \|\mathbf{w}\|_2^2
$$
4. **Match the cost**: Check which transformation gives the cost closest to $0.2$.

```python
import numpy as np  
  
# Given data  
x = np.array([-0.5, 0.39, 1.19, -1.08])  
yr = np.array([-0.86, -0.61, 1.37, 0.10])  
lambda_ = 0.25  
w0 = 0.0  
w = np.array([0.39, 0.77])  
  
# Function to standardize the design matrix  
def standardize(X):  
    mean = np.mean(X, axis=0)  
    std = np.std(X, axis=0, ddof=1)  
    return (X - mean) / std  
  
# Function to calculate the ridge regression cost  
def ridge_regression_cost(X, y, w, w0, lambda_):  
    y_pred = X.dot(w) + w0  
    error = y - y_pred  
    return np.sum(error**2) + lambda_ * np.sum(w**2)  
  
# Transformations that match the dimensions of w  
transformations = {  
    'A': lambda x: np.column_stack((x, x**3)),  
    'B': lambda x: np.column_stack((x, np.exp(x))),  
    'C': lambda x: np.column_stack((x, np.sin(x))),  
    'D': lambda x: np.column_stack((x, x**2))  
}  
  
# Iterate over transformations and calculate costs  
for key, transform in transformations.items():  
    X = transform(x)  
    X_standardized = standardize(X)  
    cost = ridge_regression_cost(X_standardized, yr, w, w0, lambda_)  
    print(f"Transformation {key}: Cost = {cost:.4f}")
```
