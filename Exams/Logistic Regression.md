
## F23, Q6) Which one of the following combination of transformation and weight vector corresponds to this classifier?

![[Pasted image 20240517234201.png]]

### Script to plot the classifier
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
  
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))  
  
  
def plot_decision_boundary(w, transform, title):  
    x1 = np.linspace(-3, 3, 300)  
    x2 = np.linspace(-3, 3, 300)  
    X1, X2 = np.meshgrid(x1, x2)  
    X_transform = transform(X1, X2)  
    # Flatten the grid to apply the dot product and then reshape it back  
    Z = sigmoid(np.dot(w, X_transform.reshape(X_transform.shape[0], -1))).reshape(X1.shape)  
  
# just different colors one with gradient one with just 2 colors  
    plt.contourf(X1, X2, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])  
    #plt.contourf(X1, X2, Z, levels=50, cmap='coolwarm', alpha=0.8)  
  
    plt.colorbar()  
    plt.title(title)  
    plt.xlabel('$b_1$')  
    plt.ylabel('$b_2$')  
    plt.show()  
  
# Transformation functions  
def transform_A(X1, X2):  
    return np.array([np.ones_like(X1), X1 ** 2, X2 ** 2])  
  
  
def transform_B(X1, X2):  
    return np.array([np.ones_like(X1), X1 ** 2, X2 ** 2])  
  
  
def transform_C(X1, X2):  
    return np.array([np.ones_like(X1), X1 ** 3, X2 ** 2])  
  
  
def transform_D(X1, X2):  
    return np.array([np.ones_like(X1), X1 ** 3, X2 ** 2])  
  
  
# Weights  
w_A = np.array([0.31, -0.06, 0.07])  
w_B = np.array([0.72, 3.13, -0.25])  
w_C = np.array([0.31, -0.06, 0.07])  
w_D = np.array([0.72, 3.13, -0.25])  
  
# Plot each option  
plot_decision_boundary(w_A, transform_A, 'Option A')  
plot_decision_boundary(w_B, transform_B, 'Option B')  
plot_decision_boundary(w_C, transform_C, 'Option C')  
plot_decision_boundary(w_D, transform_D, 'Option D')  
  
  
def evaluate_classifier(w, transform, x):  
    X_transform = transform(x[0], x[1])  
    score = np.dot(w, X_transform)  
    prob = sigmoid(score)  
    return prob  
  
# Test points  
x1 = np.array([3, 0])  
x2 = np.array([-2, 0])  
  
# Evaluate each option for both test points  
print("Evaluating Option A:")  
print(f"Test point {x1}: {evaluate_classifier(w_A, transform_A, x1)}")  
print(f"Test point {x2}: {evaluate_classifier(w_A, transform_A, x2)}\n")  
  
print("Evaluating Option B:")  
print(f"Test point {x1}: {evaluate_classifier(w_B, transform_B, x1)}")  
print(f"Test point {x2}: {evaluate_classifier(w_B, transform_B, x2)}\n")  
  
print("Evaluating Option C:")  
print(f"Test point {x1}: {evaluate_classifier(w_C, transform_C, x1)}")  
print(f"Test point {x2}: {evaluate_classifier(w_C, transform_C, x2)}\n")  
  
print("Evaluating Option D:")  
print(f"Test point {x1}: {evaluate_classifier(w_D, transform_D, x1)}")  
print(f"Test point {x2}: {evaluate_classifier(w_D, transform_D, x2)}")
```

--- 
## S23, Q18)

Suppose a logistic regression model has been trained on a two-dimensional dataset based on two attributes from the Sound Classification dataset. Now consider four test observations:

$$
\mathbf{x}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}^\top
$$

$$
\mathbf{x}_2 = \begin{bmatrix} -1.0 \\ 0.6 \end{bmatrix}^\top
$$

$$
\mathbf{x}_3 = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}^\top
$$

$$
\mathbf{x}_4 = \begin{bmatrix} -0.91 \\ -0.16 \end{bmatrix}^\top
$$

Additionally, the following probabilities are known

$$
p(y = \text{Machine} \mid \mathbf{x}_1) = 0.73
$$

$$
p(y = \text{Machine} \mid \mathbf{x}_2) = 0.47
$$

$$
p(y = \text{Machine} \mid \mathbf{x}_3) = 0.73
$$

Determine which one of the following statements is true.

*Hint: The problem can be solved without complicated numerical computation.*

A. $p(y = \text{Machine} \mid \mathbf{x}_4) < p(y = \text{Machine} \mid \mathbf{x}_2)$

B. $p(y = \text{Machine} \mid \mathbf{x}_4) = p(y = \text{Machine} \mid \mathbf{x}_1)$

C. $p(y = \text{Machine} \mid \mathbf{x}_4) > p(y = \text{Machine} \mid \mathbf{x}_2)$

D. $p(y = \text{Machine} \mid \mathbf{x}_4) > p(y = \text{Machine} \mid \mathbf{x}_3)$


### Solution

Sure, let’s go through this step by step. The problem involves determining the probability of an event (in this case, $y = \text{Machine}$) for different observations given a logistic regression model. The task is to compare these probabilities and decide which statement is true. Here's how we can approach this problem programmatically.

### Understanding Logistic Regression

Logistic regression calculates the probability of a binary event using the logistic function. For a given observation $\mathbf{x} = [x_1, x_2]$, the probability that $y = \text{Machine}$ is given by:

$$
P(y = \text{Machine} \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

where:
- $\mathbf{w}$ is the coefficient vector.
- $b$ is the intercept.

### Given Data

We are given the probabilities for some observations:

$$
\mathbf{x_1} = [1, 1] \quad P(y = \text{Machine} \mid \mathbf{x_1}) = 0.73
$$

$$
\mathbf{x_2} = [-1.0, 0.6] \quad P(y = \text{Machine} \mid \mathbf{x_2}) = 0.47
$$

$$
\mathbf{x_3} = [-0.5, -0.5] \quad P(y = \text{Machine} \mid \mathbf{x_3}) = 0.73
$$

### Goal

Determine $P(y = \text{Machine} \mid \mathbf{x_4})$ for:

$$
\mathbf{x_4} = [-0.91, -0.16]
$$

### Steps

1. **Calculate the coefficients**: Using the given probabilities and observations, we need to estimate the coefficients $\mathbf{w}$ and intercept $b$.
2. **Calculate the probability for $\mathbf{x_4}$**: Once we have the model parameters, we can calculate the probability for $\mathbf{x_4}$.
3. **Compare and decide**: Compare the probability $P(y = \text{Machine} \mid \mathbf{x_4})$ with other given probabilities to determine the correct statement.


```python
import numpy as np  
from scipy.optimize import minimize  
from sklearn.linear_model import LogisticRegression  
  
# Given data  
X = np.array([  
    [1, 1],  
    [-1.0, 0.6],  
    [-0.5, -0.5]  
])  
  
y_prob = np.array([0.73, 0.47, 0.73])  
  
# Define the logistic function  
def logistic(z):  
    return 1 / (1 + np.exp(-z))  
  
# Define the loss function for optimization  
def loss(params, X, y_prob):  
    intercept = params[0]  
    coef = params[1:]  
    z = np.dot(X, coef) + intercept  
    predictions = logistic(z)  
    return np.sum((predictions - y_prob) ** 2)  
  
# Initial guess for parameters  
initial_params = np.zeros(X.shape[1] + 1)  
  
# Optimize to find the best parameters  
result = minimize(loss, initial_params, args=(X, y_prob), method='BFGS')  
intercept, coef = result.x[0], result.x[1:]  
  
print(f"Estimated intercept: {intercept}")  
print(f"Estimated coefficients: {coef}")  
  
# Given observation x4  
x4 = np.array([-0.91, -0.16])  
  
# Calculate the probability for x4  
z4 = np.dot(x4, coef) + intercept  
p_y_machine_x4 = logistic(z4)  
print(f"P(y = Machine | x4) = {p_y_machine_x4:.2f}")  
  
# Compare and decide  
p_y_machine_x2 = y_prob[1]  
  
if p_y_machine_x4 < p_y_machine_x2:  
    print("A. P(y = Machine | x4) < P(y = Machine | x2)")  
elif np.isclose(p_y_machine_x4, y_prob[0]):  
    print("B. P(y = Machine | x4) = P(y = Machine | x1)")  
elif p_y_machine_x4 > p_y_machine_x2:  
    print("C. P(y = Machine | x4) > P(y = Machine | x2)")  
else:  
    print("E. Don’t know")
```
