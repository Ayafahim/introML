#PCA
## Covariance Matrix, Eigen Decomposition, and Selecting Principal Components

### Covariance Matrix

The **covariance matrix** ($\Sigma$) captures the covariance between every pair of dimensions in your dataset, with the variance of each dimension on its diagonal.

#### Definition
For a data matrix $X$ with $n$ features and $m$ observations:

$$
\Sigma = \frac{1}{m-1} X^T X
$$

Where:
- $X^T$ is the transpose of $X$,
- Each entry $\Sigma_{ij}$ represents the covariance between the $i$-th and $j$-th dimension.

#### Interpretation
- Diagonal elements ($\Sigma_{ii}$) represent the variance of the $i$-th dimension.
- Off-diagonal elements ($\Sigma_{ij}$) represent the covariance between dimensions.

### Eigen Decomposition

Eigen decomposition involves decomposing the covariance matrix into its eigenvectors (directions of maximum variance) and eigenvalues (magnitude of the variance in these directions).

#### Mathematical Background

Given the covariance matrix $\Sigma$, find $V$ (eigenvectors) and $\Lambda$ (eigenvalues) such that:

$$
\Sigma V = V \Lambda
$$

#### Interpretation
- Eigenvectors ($V$) define the axes of the new feature space.
- Eigenvalues ($\Lambda$) indicate the amount of variance carried in each principal component direction.

### Selecting Principal Components

Principal components are selected based on the eigenvalues from the eigen decomposition of the covariance matrix.

#### Process
1. **Sort Eigenvalues and Eigenvectors**:
   Sort the eigenvalues in descending order and reorder the eigenvectors accordingly.
   
2. **Choose Top Components**:
   Select the top $k$ eigenvectors as principal components.

#### Explained Variance
The explained variance ratio for each component is given by the proportion of the eigenvalue relative to the sum of all eigenvalues:

$$
\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^n \lambda_j}
$$

Where $\lambda_i$ is the $i$-th eigenvalue.

---
## Understanding the Matrix $V$ in PCA

The matrix $V$ in the context of Principal Component Analysis (PCA) is often referred to as the matrix of right singular vectors if PCA is performed via Singular Value Decomposition (SVD), or it can be the matrix of eigenvectors if PCA is done directly via eigen decomposition of the covariance matrix. Each column of $V$ represents a principal component of the data.

### Structure of Matrix $V$

$V$ is typically structured as follows:

$$
V = \begin{bmatrix}
v_{11} & v_{12} & \cdots & v_{1p} \\
v_{21} & v_{22} & \cdots & v_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
v_{n1} & v_{n2} & \cdots & v_{np}
\end{bmatrix}
$$

Where:
- Each row corresponds to an original feature/dimension of the data.
- Each column represents a principal component, which is a new axis in the transformed feature space.

### Columns: Principal Components

- **First Column ($v_1$)**: This column represents the first principal component. It's the direction in which the data varies the most. In other words, if you project the data onto this axis, you'll see the largest spread of data, or the most variance.
- **Second Column ($v_2$)**: Represents the second principal component, which is orthogonal (at a right angle) to the first. It accounts for the next highest variance in the dataset that hasn't been captured by the first principal component.
- **Subsequent Columns**: Each subsequent column captures progressively less variance in the data.

### Rows: Contribution of Each Original Feature

- **Each Row in a Column**: Represents how much each original feature contributes to the corresponding principal component. For example, in the first column ($v_1$), the value at the first row tells you how much the first feature contributes to the first principal component.

### Interpretation with an Example

Imagine we have a dataset with three features (X, Y, Z) and we calculate PCA, resulting in a matrix $V$ like this:

$$
V = \begin{bmatrix}
0.5 & -0.1 & 0.2 \\
0.4 & 0.9 & -0.1 \\
0.7 & -0.4 & 0.9
\end{bmatrix}
$$

#### How to Interpret:
- **First Column**: The values (0.5, 0.4, 0.7) suggest that all three features contribute positively to the first principal component, with the third feature (Z) having the highest influence.
- **Second Column**: Shows a high positive contribution from the second feature (Y) and negative contributions from X and Z to the second principal component. This axis is orthogonal to the first, providing unique information not captured by the first component.
- **Third Column**: Indicates another distinct pattern of data variance, mostly dominated by the third feature (Z).

### Using Matrix $V$ for Data Projection

To project data onto these new axes defined by the principal components:

- **Projection**: Multiply your original data matrix by $V$. This operation transforms the data from the original feature space to the new space defined by the principal components.

### Interpretation of Projections onto Principal Components


- **Magnitude**: Indicates the strength or influence of the feature on the principal component. Larger absolute values mean greater influence.
- **Sign**: Indicates the direction of the relationship between the feature and the component. A positive sign means that as the feature increases, the projection on the component increases; a negative sign indicates the opposite.


When interpreting projections:

- **Low Value of Feature $x$**: 
  - If the coefficient of $x$ in $V$ is negative, a numerically low value (which might be negative after centering and scaling) will contribute positively due to the multiplication of two negatives (negative times negative equals positive).
  - Conversely, if the coefficient is positive, a low value will contribute negatively.

- **High Value of Feature $x$**: 
  - If the coefficient of $x$ in $V$ is positive, a high value will contribute positively.
  - If the coefficient is negative, a high value will contribute negatively.

### Table Overview

| Condition  | Coefficient Sign | Contribution to PC |
| ---------- | ---------------- | ------------------ |
| Low Value  | Positive         | Negative           |
| Low Value  | Negative         | Positive           |
| High Value | Positive         | Positive           |
| High Value | Negative         | Negative           |

### Example Scenario

Suppose we have a matrix $V$ and an observation vector $x$:

$$
V = \begin{bmatrix}
0.9 & -0.4 \\
-0.4 & 0.9
\end{bmatrix}
$$

And:

$$
x = \begin{bmatrix}
1.5 \\ 
-2
\end{bmatrix}
$$

#### Interpreting $x$ in Terms of $V$:

1. **First Component**: $x_1 = 1.5$, $x_2 = -2$
   - $v_{11} = 0.9$ (positive), so high $x_1$ contributes positively.
   - $v_{21} = -0.4$ (negative), and $x_2$ is low (negative), which also contributes positively (negative times negative equals positive).
   
2. **Second Component**: 
   - $v_{12} = -0.4$ (negative), so high $x_1$ contributes negatively.
   - $v_{22} = 0.9$ (positive), and $x_2$ is low, which contributes negatively (positive times negative equals negative).

### Conclusion

- **Projection on First Component**: Strong positive contribution overall.
- **Projection on Second Component**: Strong negative contribution.


---
## Understanding the Matrix $S$ in PCA via SVD

### Structure of Matrix $S$

Matrix $S$ is a diagonal matrix with the singular values arranged in descending order. The structure of $S$ is as follows:

$$
S = \begin{bmatrix}
\sigma_1 & 0 & 0 & \cdots & 0 \\
0 & \sigma_2 & 0 & \cdots & 0 \\
0 & 0 & \sigma_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \sigma_p
\end{bmatrix}
$$

Where:
- $\sigma_1, \sigma_2, \sigma_3, \ldots, \sigma_p$ are the singular values.
- $p$ is the number of singular values, which is typically equal to the number of dimensions of the input data or the number of principal components, depending on the context.
### Interpretation with an Example

Suppose we perform SVD on a dataset and obtain the following $S$ matrix:

$$
S = \begin{bmatrix}
5 & 0 & 0 \\
0 & 3 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

#### How to Interpret:
- **First Singular Value (5)**: Indicates that the first principal component captures a large variance. Specifically, $5^2 = 25$ units of variance.
- **Second Singular Value (3)**: The second principal component captures $3^2 = 9$ units of variance.
- **Third Singular Value (1)**: The third principal component captures only $1^2 = 1$ unit of variance, significantly less than the first two components.


### Singular Values: Explained Variance

Each singular value $\sigma_i$ in $S$ represents the square root of an eigenvalue from the covariance matrix of the dataset, which in turn is related to the variance captured by the corresponding principal component:

- **Larger Singular Values**: A larger singular value indicates that the corresponding principal component captures a significant amount of variance in the dataset.
- **Decreasing Order**: Singular values are always arranged in decreasing order in $S$. This order determines the importance of the principal components; the first few components are the most significant.
### Using Matrix $S$ for Data Scaling

When transforming data using the principal components:

- **Scaling by $S$**: Each principal component vector in $V$ can be scaled by the corresponding singular value in $S$. This operation emphasizes the importance of the components based on the variance they capture.

---
## Example 1: Interpreting PCA Output from SVD 

![[Pasted image 20240509145810.png]]

Given the matrices $S$ and $V$ from the Singular Value Decomposition (SVD) of a standardized data matrix $\hat{X}$:

### S Matrix (Singular Values)
$$
S = \begin{bmatrix}
13.5 & 0 & 0 & 0 & 0 & 0 \\
0 & 7.6 & 0 & 0 & 0 & 0 \\
0 & 0 & 6.5 & 0 & 0 & 0 \\
0 & 0 & 0 & 5.8 & 0 & 0 \\
0 & 0 & 0 & 0 & 3.5 & 0 \\
0 & 0 & 0 & 0 & 0 & 2.0
\end{bmatrix}
$$

### V Matrix (Right Singular Vectors)
$$
V = \begin{bmatrix}
0.38 & -0.51 & 0.23 & 0.47 & -0.55 & 0.11 \\
0.41 & 0.41 & -0.53 & 0.24 & 0.00 & 0.58 \\
0.50 & 0.34 & -0.13 & 0.15 & -0.05 & -0.77 \\
0.29 & 0.48 & 0.78 & -0.17 & 0.00 & 0.23 \\
0.45 & -0.42 & 0.09 & 0.03 & 0.78 & 0.04 \\
0.39 & -0.23 & -0.20 & -0.82 & -0.30 & 0.04
\end{bmatrix}
$$
## Step-by-Step Process for Using SVD Output in PCA

### Step 1: Understand the SVD Output

From the Singular Value Decomposition (SVD) of your data matrix $\hat{X}$, you have the matrices $S$ and $V$.

- **Matrix $S$**: Contains the singular values. For PCA, these singular values are crucial because the square of these values will give you the eigenvalues used in PCA.
- **Matrix $V$**: Contains the right singular vectors, which represent the principal components in the PCA context.

### Step 2: Calculate the Explained Variance

The percentage of variance explained by each principal component is computed from the eigenvalues, which are the squares of the singular values in matrix $S$.

#### Eigenvalues ($\lambda_i$):
$$
\lambda = [13.5^2, 7.6^2, 6.5^2, 5.8^2, 3.5^2, 2.0^2]
$$
$$
\lambda = [182.25, 57.76, 42.25, 33.64, 12.25, 4.00]
$$

#### Total Variance:
$$
\text{Total Variance} = \sum \lambda = 182.25 + 57.76 + 42.25 + 33.64 + 12.25 + 4.00 = 332.15
$$

#### Explained Variance Ratio for each component:
$$
\text{Explained Variance Ratio} = \frac{\lambda_i}{\text{Total Variance}}
$$

- For $\lambda_1 = 182.25$:
$$
\frac{182.25}{332.15} \approx 0.549
$$
- For $\lambda_2 = 57.76$:
$$
\frac{57.76}{332.15} \approx 0.174
$$

#### Cumulative Explained Variance:
Calculate the cumulative sum of the explained variance ratios as you add more principal components.

### Step 3: Match the Cumulative Explained Variance to the Curves

Based on the cumulative sums calculated:

- $k=1$: Approximately 55%
- $k=2$: Approximately 72% (55% + 17%) 
- $k=3$: Approximately 85% (72% + 13%)

Remember this also means that the first to components account for 72% of the variance in the data, PC1 for 55% & PC2 for 17%, The first 3 for 85% and so on.

#### Example Analysis for Curve Matching

- At $k=1$, around 55%
- At $k=2$, around 72%

Now, examine each curve in Figure 2:

- **Curve 1**: Starts near 0.55, reaches about 0.72 by $k=2$, and continues in a pattern similar to your calculations.
- **Curve 2**: The starting value or slope might not match.
- **Curve 3 & 4**: Similarly evaluate based on how the initial and subsequent values align with your calculations.

It appears **Curve 1** is likely the match based on the provided numbers, but you should calculate the exact percentages for all components to confirm.

```python
import numpy as np  
import matplotlib.pyplot as plt  
  
# Singular values from matrix S  
singular_values = np.array([13.5, 7.6, 6.5, 5.8, 3.5, 2.0])  
  
# Step 1: Calculate eigenvalues from the singular values (squared)  
eigenvalues = singular_values**2  
  
# Step 2: Calculate total variance  
total_variance = np.sum(eigenvalues)  
  
# Step 3: Calculate explained variance ratio for each principal component  
explained_variance_ratio = eigenvalues / total_variance  
  
# Step 4: Calculate cumulative explained variance  
cumulative_explained_variance = np.cumsum(explained_variance_ratio)  
  
# Printing explained variances  
print("Explained Variance Ratios:", explained_variance_ratio)  
print("Cumulative Explained Variance:", cumulative_explained_variance)  
  
# Step 5: Plotting  
plt.figure(figsize=(8, 6))  
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')  
plt.xlabel('Number of Components')  
plt.ylabel('Cumulative Explained Variance')  
plt.title('Explained Variance by Different Principal Components')  
plt.grid(True)  
plt.xticks(range(1, len(explained_variance_ratio) + 1))  
plt.axhline(y=0.72, color='r', linestyle='--', label='72% Explained Variance')  
plt.axhline(y=0.85, color='g', linestyle='--', label='85% Explained Variance')  
plt.legend()  
plt.show()
```

### Result
```
Explained Variance Ratios: [0.54869788 0.17389734 0.12720157 0.10127954 0.03688093 0.01204275]
Cumulative Explained Variance: [0.54869788 0.72259521 0.84979678 0.95107632 0.98795725 1.        ]
```

If you want to add conditions to check the different components

```python
# Conditions based on the explained variance calculations
condition_a = explained_variance_ratio[0] > 0.55
condition_b = cumulative_explained_variance[2] > 0.90  # Index 2 because it's the third element (0-based index)
condition_c = explained_variance_ratio[-1] > 0.01 #The last principal component
condition_d = cumulative_explained_variance[4] > 0.95  # Index 4 for the fifth element

# Print results
print("Condition A (First PC > 55%):", condition_a)
print("Condition B (First three PCs > 90%):", condition_b)
print("Condition C (Last PC > 1%):", condition_c)
print("Condition D (First five PCs > 95%):", condition_d)
```

OR use this

```python
import numpy as np  
  
# Singular values from the matrix S  
singular_values = np.array([43.4, 23.39, 18.26, 9.34, 2.14])  
  
# Square the singular values to get the variances  
variances = singular_values ** 2  
  
# Total variance is the sum of individual variances  
total_variance = np.sum(variances)  
  
# Proportion of variance explained by each principal component  
proportion_variance_explained = variances / total_variance  
  
# Compute cumulative variance explained by the first one, two, three, and four principal components  
cumulative_variance_one = np.sum(proportion_variance_explained[:1])  
cumulative_variance_three = np.sum(proportion_variance_explained[:3])  
cumulative_variance_four = np.sum(proportion_variance_explained[:4])  
  
# Cumulative variance explained by the last four principal components (excluding the first one)  
cumulative_variance_last_four = np.sum(proportion_variance_explained[1:])  
  
# Print the results  
print("Proportion of variance explained by each component:", proportion_variance_explained)  
print("Cumulative variance explained by one component:", cumulative_variance_one)  
print("Cumulative variance explained by three components:", cumulative_variance_three)  
print("Cumulative variance explained by four components:", cumulative_variance_four)  
print("Cumulative variance explained by the last four components:", cumulative_variance_last_four)  
  
# Evaluate the statements given in the problem  
A = cumulative_variance_last_four < 0.3  
B = cumulative_variance_three > 0.9  
C = cumulative_variance_four < 0.95  
D = cumulative_variance_one > 0.715  
  
# Print the truth value of each statement  
print("Statement A is", A)  
print("Statement B is", B)  
print("Statement C is", C)  
print("Statement D is", D)  
  
# Identify the correct statement  
statements = [A, B, C, D]  
statement_labels = ['A', 'B', 'C', 'D']  
for i, statement in enumerate(statements):  
    if statement:  
        print(f"The correct statement is: {statement_labels[i]}")
```


---

## Projection of an Observation onto Principal Components

#### Question 3, Spring 18
According to the extracted PCA directions given by the matrix V in the above what will be the coordinate of the standardized observation $x^* = [-0.1, 0.2, 0.1, -0.3, 1, 0.5]$ when projected onto  
the first two principal components?

Given an observation vector $x^*$ and a matrix $V$ containing principal component directions, the projection of $x^*$ onto the principal components involves calculating the dot products of $x^*$ with each of the principal component vectors.

### Given:

- **Observation Vector**: $x^* = [-0.1, 0.2, 0.1, -0.3, 1, 0.5]$
- **Principal Component Matrix $V$**: (first two columns are used)
  $$
  V = \begin{bmatrix}
  0.38 & -0.51 \\
  0.41 & 0.41 \\
  0.50 & 0.34 \\
  0.29 & 0.48 \\
  0.45 & -0.42 \\
  0.39 & -0.23
  \end{bmatrix}
  $$

### Steps to Compute the Projection:

1. **Extract the Principal Components**:
   - Let $v_1$ be the first column of $V$ and $v_2$ be the second column.

2. **Calculate the Projection**:
   - The projection of $x^*$ onto $v_1$ is calculated as:
     $$
     p_1 = x^* \cdot v_1 = [-0.1, 0.2, 0.1, -0.3, 1, 0.5] \cdot [0.38, 0.41, 0.50, 0.29, 0.45, 0.39]
     $$
   - Similarly, the projection onto $v_2$:
     $$
     p_2 = x^* \cdot v_2 = [-0.1, 0.2, 0.1, -0.3, 1, 0.5] \cdot [-0.51, 0.41, 0.34, 0.48, -0.42, -0.23]
     $$

3. **Projection Coordinates**:
   - The coordinates of the projection in the subspace defined by the first two principal components are $(p_1, p_2)$.

### Python Implementation:

```python
import numpy as np

# Principal Component Matrix V (example subset)
V = np.array([
    [0.38, -0.51],
    [0.41, 0.41],
    [0.50, 0.34],
    [0.29, 0.48],
    [0.45, -0.42],
    [0.39, -0.23]
])

# Observation vector x*
x_star = np.array([-0.1, 0.2, 0.1, -0.3, 1, 0.5])

# Calculate projections
p1 = np.dot(x_star, V[:, 0])  # Projection on v1
p2 = np.dot(x_star, V[:, 1])  # Projection on v2

# Print projection coordinates
print(f"Projection Coordinates: ({p1:.3f}, {p2:.3f})")
```

### Result
`Projection Coordinates: (0.652, -0.512)`

---
## Understanding the Frobenius Norm and Singular Value Decomposition (SVD)

### Frobenius Norm

The Frobenius norm of a matrix, denoted as $||\tilde{X}||_F$, is a measure of the "size" or "magnitude" of the matrix. It is defined as the square root of the sum of the absolute squares of its elements. For a matrix with singular values $\sigma_1, \sigma_2, ..., \sigma_n$, the squared Frobenius norm can be expressed as:

$$
||\tilde{X}||_F^2 = \sigma_1^2 + \sigma_2^2 + \sigma_3^2 + \ldots + \sigma_n^2
$$

This means that the squared Frobenius norm is the sum of the squares of all its singular values.

### Singular Value Decomposition (SVD)

SVD is a method of decomposing a matrix into three other matrices, such that:

$$
\tilde{X} = U \Sigma V^T
$$

- $U$ and $V$ are orthogonal matrices (where $U^TU = I$ and $V^TV = I$).
- $\Sigma$ is a diagonal matrix containing the singular values of $\tilde{X}$.

SVD is particularly useful in many applications, including PCA, because it helps in identifying the directions of maximum variance in data.

### Simple Example

Consider a $2 \times 2$ matrix:

$$
A = \begin{bmatrix}
3 & 4 \\
5 & 2
\end{bmatrix}
$$

After applying SVD, we might find:

$$
U = \begin{bmatrix}
0.6 & -0.8 \\
0.8 & 0.6
\end{bmatrix}, \quad
\Sigma = \begin{bmatrix}
7 & 0 \\
0 & 2
\end{bmatrix}, \quad
V^T = \begin{bmatrix}
0.8 & 0.6 \\
-0.6 & 0.8
\end{bmatrix}
$$

Where the diagonal elements of $\Sigma$ are the singular values, indicating the scales of the new axes found by SVD.






