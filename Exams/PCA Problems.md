## F21, Q3) Which one of the following statements is true? #ExplainedVar

A. The variance explained by the last four principal components is less than 0.3 of the total variance.

B. The variance explained by the first three principal components is greater than 0.9 of the total variance.

C. The variance explained by the first four principal components is less than 0.95 of the total variance.

D. The variance explained by the first principal component is greater than 0.715 of the total variance.

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


## F21, Q4) Consider again the PCA analysis for the Olive Oil dataset, in particular the SVD decomposition of $\mathbf{\tilde{X}}$ in Equation (1). Which one of the following statements is true?

A. An observation with a low value of $x_1$ (palmitic), a low value of $x_2$ (palmitoleic), a high value of $x_4$ (oleic), and a low value of $x_5$ (linoleic) will typically have a negative value of the projection onto principal component number 1. (Correct Answer)

B. An observation with a high value of $x_3$ (stearic) will typically have a negative value of the projection onto principal component number 2.

C. An observation with a low value of $x_1$ (palmitic), a high value of $x_2$ (palmitoleic), and a high value of $x_4$ (oleic) will typically have a positive value of the projection onto principal component number 4.

D. An observation with a low value of $x_1$ (palmitic), a low value of $x_2$ (palmitoleic), and a high value of $x_5$ (linoleic) will typically have a negative value of the projection onto principal component number 3.


### Steps to Solve the Problem:

> [!TIP]
> Refer to the table overview of projections in [[Chapters/PCA|PCA]]

1. **Examine the $V$ Matrix**: Look at the coefficients (entries) for each principal component (column in $V$). Positive coefficients indicate a positive correlation with the principal component and negative coefficients indicate a negative correlation.
2. **Consider the Sign and Magnitude of Coefficients**: The magnitude tells how much influence that variable has on the principal component. The sign (+/-) indicates the direction of the influence.
3. **Analyze Given Statements**:
   - For a statement that says "an observation with a low/high value of $x$ will have a positive/negative projection," you should:
     - **Low Value of $x$**: If $x$ has a negative coefficient in a principal component, a low value (which is numerically negative after centering and scaling) contributes positively to that component because of the multiplication of two negatives.
     - **High Value of $x$**: If $x$ has a positive coefficient in a principal component, a high value contributes positively to that component.

### Evaluating Statement A:
- **Statement A**: An observation with a low value of \( x1 \) (palmitic), a low value of \( x2 \) (palmitoleic), a high value of \( x4 \) (oleic), and a low value of \( x5 \) (linoleic) will typically have a negative value of the projection onto principal component number 1.
### Steps:
1. **Identify the First Principal Component**:
   - Look at the **first column** of the \( V \) matrix because each column corresponds to one principal component.

2. **Check the Coefficients**:
   - **For \( x1 \) and \( x2 \)**: Since you're given "low" values for these, and assuming these coefficients are positive in the first principal component, the negative signs of these "low" values (after mean subtraction and scaling) would contribute negatively to the projection.
   - **For \( x4 \)**: With a "high" value and assuming a positive coefficient, this would contribute positively.
   - **For \( x5 \)**: A "low" value here would contribute negatively if the coefficient is positive.

3. **Determine the Net Effect**:
   - If the first principal component's coefficients for \( x1 \) and \( x2 \) are positive, their "low" values contribute negatively.
   - If \( x4 \) has a positive coefficient, its "high" value contributes positively.
   - If \( x5 \) has a positive coefficient, its "low" value also contributes negatively.
   - Sum these influences (considering their signs). The predominant sign of the resulting sum will indicate whether the projection is generally positive or negative.

### Conclusion:
If the negative contributions from the "low" values of \( x1 \), \( x2 \), and \( x5 \) outweigh the positive contribution from the "high" value of \( x4 \), the overall projection onto the first principal component would be negative, making Statement A true if this is the case. However, the actual truth of the statement depends on the specific coefficients in the \( V \) matrix for the first principal component.

This approach lets you use the properties of the \( V \) matrix effectively to predict how changes in the original variables influence their representation in the transformed feature space created by PCA.


## F21, Q5) All the objects from four regions of origin are projected onto the first two principal components and visualised as a scatter plot in Figure 4. Which one of the following statements is true?

![[Pasted image 20240508203148.png]]

A Principal Component Analysis (PCA) is carried out on all the eight attributes of the Olive Oil dataset in Table 1. All the objects from four regions of origin are projected onto the first two principal components and visualised as a scatter plot in Figure 4. Which one of the following statements is true?

A. There exists a logistic regression classifier that takes the observations projected onto the first two principal components as input, which can binary classify the observations in the two regions South Apulia (y = 3) and Sicily (y = 4) with 0 error.

B. Any classifications tree using axis-aligned splits that takes the observation projected onto the first two principal components as input and binary classify the observations in the two regions South Apulia (y = 3) and Umbria (y = 9) has an error strictly greater than 0.

C. Any classification tree using axis-aligned splits that takes all eight attributes as input and binary classify the observations in the two regions South Apulia (y = 3) and Inner Sardinia (y = 5) has an error strictly greater than 0.

D. There exists a logistic regression classifier that takes all eight attributes as input, which can binary classify the observations in the two regions South Apulia (y = 3) and Umbria (y = 9) with 0 error.

### Explanation of Answers

- **Answer A** is incorrect, since the points of the two classes South Apulia (y = 3) and Sicily (y = 4) are not linearly separable in Figure 4.
- **Answer B** is incorrect, since a tree with two leafs (splitting e.g. around −1 in the projection onto the first principal component) will be able to perfectly classify the objects.
- **Answer C** is incorrect, since a classification tree is always able to obtain an error of 0 when there is no identical training object in the two classes (unless the tree complexity is limited).
- **Answer D** is correct, since the two classes South Apulia (y = 3) and Umbria (y = 9) are linearly separable in the PCA plot. Furthermore, if points are linearly separable in the projection onto the first two principal components, then they are also linearly separable in the original attribute space.


## S16, Q3) Plot of each observation plotted onto the two first principal directions given in Equation...Which of the following statements best describes the development of  the measurements? 

Full question:
Consider again the Occupancy dataset of Table 1. A plot of each observation plotted onto the two first principal directions given in Equation (2) is shown in Figure 3. As seen in the plot, the observations are made successively over time and therefore the room measurements form ”trajectories”. Suppose a rooms measurements starts at the black circle and ends a few hours later at the blue square. Which of the following statements best describes the development of  
the measurements?

![[Screenshot 2024-04-18 at 17.30.27.png]]
![[Pasted image 20240418173502.png]]
![[Screenshot 2024-04-18 at 17.30.42.png]]

The matrix `S` is a diagonal matrix containing the singular values of the PCA, and the matrix `V` contains the principal directions (eigenvectors). We have to only look at the mesaurements from start to end which goes from 1 to -3 on the y-axis. The operation `-3v2 - 1v2` (it is -1 because we compute difference) is linear combination of the second column of the `V` matrix, which corresponds to the second principal component. 

```python
import numpy as np  
  
# V matrix from the second image  
V = np.array([  
    [-0.3, -0.5,  0.7,  0.2,  0.2],  
    [-0.4,  0.6, -0.0,  0.2,  0.7],  
    [-0.4, -0.4, -0.7,  0.4, -0.0],  
    [-0.6, -0.1, -0.1, -0.8,  0.1],  
    [-0.5,  0.4,  0.2,  0.2, -0.7]  
])  
  
# Extracting the second principal component (v2)  
v2 = V[:, 1]  # This is the second column of V  
  
# Computing -3v2 - 1v2  
result_vector = -3 * v2 - 1 * v2  
  
# Display the result  
print(result_vector)


#result
#[ 2.  -2.4  1.6  0.4 -1.6]
# 2 = x1, -2.4 = x2, etc...
#If negative coefficent negative impact, otherwise positive impact

# We see the temperature goes up, the humidity drops and the light goes up, therefore option A is correct.
```

## S18, Q2) In Figure 2 is given the pct. of variance explained by retaining the first k principal components as a function of k. Which one of the four curves corresponds to the correct curve of variance explained as function of the number of principal components retained? 

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

### Calculate the Explained Variance

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

### Match the Cumulative Explained Variance to the Curves

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

### Script to calculate and plot explained variance.

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



## S18, Q3) According to the extracted PCA directions given by the matrix V in the above what will be the coordinate of the standardized observation $x^* = [-0.1, 0.2, 0.1, -0.3, 1, 0.5]$ when projected onto the first two principal components?

Given an observation vector $x^*$ and a matrix $V$ containing principal component directions, the projection of $x^*$ onto the principal components involves calculating the dot products of $x^*$ with each of the principal component vectors.

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


## F16, Q3) The data projected onto the two first principal components (as defined in Question 2) is given in Figure 2 where each class is indicated using different markers and colors. which one of the following statements pertaining to the PCA is correct?

![[Pasted image 20240509173452.png]]
![[Pasted image 20240509173658.png]]

![[Pasted image 20240509173716.png]]


A. A relatively long and narrow seed kernel will provide a large positive projection onto the second principal component.  

B. The first principal component pertains to the general size of seeds.  

C. A seed that has relatively small area and perimeter but large length and width of kernel will have a negative projection onto the third principal component.  

D. As the third and fourth principal components account for a low amount of the variance in the data this is a difficult classification task.

To clarify and understand the problem presented in the images you've shared, let's break down the question and solution provided:

Based on the given $V$ matrix:
- $V_1 = [-0.51, -0.51, -0.49, -0.49]$ 
- $V_2 = [0.11, -0.13, -0.69, 0.71]$

- **Statement A**: A relatively long and narrow seed kernel will provide a large positive projection onto the second principal component.
    - **Analysis**: Considering the coefficients of $V_2$ where the length and width would likely be represented by the third and fourth coefficients (assuming these dimensions), a negative coefficient for length and a positive for width in $V_2$ would suggest that an increase in width and a decrease in length contribute to a higher projection onto the second PC. Since this is the opposite of "long and narrow", statement A is likely incorrect.
  
- **Statement B**: The first principal component pertains to the general size of seeds.
    - **Analysis**: Given the roughly equal and negative weights across all features in $V_1$, it suggests that this component might indeed represent a general size factor where larger values in any feature result in a more negative value of the first principal component. This is the most plausible statement given the composition of $V_1$.

- **Statement C** and others can be analyzed similarly by comparing the directions and magnitudes of the coefficients in the $V$ matrix relative to the nature of the features (e.g., length, width).

### Conclusion
From the analysis, **Statement B** is correct as it aligns with the interpretation of the weights in $V_1$, where all features equally contribute to a single factor, likely representing the overall size or scale of the seeds.

### Plot Interpretation
Looking at the scatter plot in Figure 2, we see the data clustered by type (Kama, Rosa, Canadian) and spread along the axes of PCA1 and PCA2. This visualization helps confirm that PCA1 and PCA2 capture distinct variances which possibly correlate with physical seed characteristics like size and shape, consistent with the discussions above.


