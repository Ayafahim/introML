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



