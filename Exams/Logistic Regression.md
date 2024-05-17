
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
