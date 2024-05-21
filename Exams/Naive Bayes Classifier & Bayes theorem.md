## S19, Q13) what is then the probability it has average rating (y = 2) according to the Naıve-Bayes classifier?
![[Pasted image 20240515231345.png]]

![[Pasted image 20240515231403.png]]

### Solution
Remember that:

$$
p_{NB}(y = 2 \mid f_2 = 0, f_4 = 1, f_5 = 0) = \frac{p(f_2 = 0 \mid y = 2) p(f_4 = 1 \mid y = 2) p(f_5 = 0 \mid y = 2) p(y = 2)}{\sum_{j=1}^3 p(f_2 = 0 \mid y = j) p(f_4 = 1 \mid y = j) p(f_5 = 0 \mid y = j) p(y = j)}
$$
This:
$$
{\sum_{j=1}^3 p(f_2 = 0 \mid y = j) p(f_4 = 1 \mid y = j) p(f_5 = 0 \mid y = j) p(y = j)}
$$
Means to multiply all times where f2=0 , f4 = 1 and f5 = 0 and then also amount of classes for y  for all the classes and sum them.
So fx:


$$
= \frac{\frac{2 \times 2 \times 2 \times 3}{3 \times 3 \times 3 \times 10}}{\frac{2 \times 1 \times 2 \times 2}{2 \times 2 \times 2 \times 10} + \frac{2 \times 2 \times 2\times 3}{3 \times 3\times 3 \times 10} + \frac{4 \times 3 \times 1\times 5}{5 \times 5\times 5 \times 10}} =  \frac{533}{200}
$$
 
${p(f_2 = 0 \mid y = 2)} = 2/3$ times if we look at the table 
${p(f_4 = 1 \mid y = 2)} = 2/3$ times if we look at the table 
${(y = 2)} = 3/10$ because there are 3 classes where y = 2

This is done for all of them.

calculated in maple
```maple
2/3*2/3*2/3*3/10/(2/2*1/2*2/2*2/10 + 2/3*2/3*2/3*3/10 + 4/5*3/5*1/5*5/10)
```

--- 
## S19, Q20) Using this, what is then the probability an observation had poor rating given that ˆx2 = 0 and ˆx3 = 1?

![[Pasted image 20240516030011.png]]
![[Pasted image 20240516030016.png]]

### Solution

Remember that:

$$
p(y = 1 \mid x_2 = 0, x_3 = 1) = \frac{p(x_2 = 0, x_3 = 1 \mid y = 1)p(y = 1) }{\sum_{j=1}^3 p(x_2 = 0 \mid y = j) p(x_3 = 1 \mid y = j)p(y = j)}
$$

So:

$$
p(y = 1 \mid x_2 = 0, x_3 = 1) = \frac{0.17 \times 0.268}{0.28 \times 0.366 + 0.17\times 0.268 + 0.33 \times 0.365} = 0.17
$$

---
## F31, Q11) Consider the Table 4 that shows the class conditional joint probability for the attributes $x_1, x_3$ of the CCPP dataset after binarization, while the prior probabilities for the two classes are $p(y =  Low) = 0.53$ and $p(y = High) = 0.47$.  What is the probability the energy production to be High when $x_1 = 0$?

![[Pasted image 20240518004150.png]]

### Solution 
You just have to first take $p(x_1 = 0 | y = high)$  so this means for both $p(x_1 = 0 | x_3 = 0)$ and $p(x_1 = 0 | x_3 = 1)$

so:
$$
p(x_1 = 0 | y = high) = p(x_1 = 0 | x_3 = 1) + p(x_1 = 0 | x_3 = 0)
$$ $$
p(x_1 = 0 | y = high) = 0.25 + 0.68
$$
and for y = low:
$$
p(x_1 = 0 | y = low) = p(x_1 = 0 | x_3 = 1) + p(x_1 = 0 | x_3 = 0)
$$
so:
$$
\frac{0.47 \times (0.25 + 0.68)} {(0.47 \times (0.25 + 0.68) + 0.53 \times (0.04 + 0.03))} = 0.92
$$
#### Solution from exam pdf
![[Pasted image 20240518004947.png]]

---

## S23, Q10) What is then the probability y corresponds to Machine given an observation has ˆx2 = 1 ? With table

> [!IMPORTANT]
> The table below is different and you need to use the product rule for this. Note it says 
> p(ˆx2, ˆx4, y) og ikke p(ˆx2, ˆx4 | y)

![[Pasted image 20240520210327.png]]


So:

$p(x_2 = 1, y = machine) = 0.08 +0.01 = 0.18$
$p(x_a) = 0.08+0.1+0.16+0.16 = 0.5$


$$
\frac{(0.08 +0.1)}{(0.08+0.1+0.16+ 0.16)} = 0.36
$$
---

## S23, Q11) What is the minimum number of evaluations of the normal density function N (x|μ, σ2) we have to perform to compute this quantity?

![[Pasted image 20240520210947.png]]
---
## F23, Q11) Consider the Table 4 that shows the class conditional joint probability for the attributes x1, x3 of the CCPP dataset after binarization, while the prior probabilities for the two classes are p(y = Low) = 0.53 and p(y = High) = 0.47.  What is the probability the energy production to be High when x1 = 0?

![[Pasted image 20240520214908.png]]

$$
\frac{(0.25+0.68) * 0.47}{0.25+0.68) * 0.47 + 0.04 + 0.03) * 0.53}
$$


![[Pasted image 20240520214911.png]]

---
## F23, Q13) Multivariate normal distributions

![[Pasted image 20240520220858.png]]


## Solution

![[Pasted image 20240520221015.png]]


Consider the class conditional distributions for the $( x_1, x_3 )$ attributes of the CCPP dataset, which we model with the following multivariate normal distributions:

$$ p(x_1, x_3 \mid y = 1) = \mathcal{N}(\mu^{(1)}, \Sigma^{(1)}), \text{ where} $$

$$ \mu^{(1)} = \begin{bmatrix} 0.77 \\ -0.41 \end{bmatrix}, \quad \Sigma^{(1)} = \begin{bmatrix} 0.29 & -0.12 \\ -0.12 & 0.55 \end{bmatrix} $$

$$ p(x_1, x_3 \mid y = 2) = \mathcal{N}(\mu^{(2)}, \Sigma^{(2)}), \text{ where} $$

$$ \mu^{(2)} = \begin{bmatrix} -0.91 \\ 0.5 \end{bmatrix}, \quad \Sigma^{(2)} = \begin{bmatrix} 0.32 & -0.11 \\ -0.11 & 1.12 \end{bmatrix} $$

The prior probabilities for the classes Low and High are $( p(y = 1) = 0.53 )$ and $( p(y = 2) = 0.47 )$, respectively. We are given a new test point $( x_{\text{test}} = \begin{bmatrix} 0 \\ 0.7 \end{bmatrix} )$ to classify, and we consider a Naive Bayes approach.

### Steps to Solve the Problem:
1. **Calculate the class-conditional probabilities $( p(x_{\text{test}} \mid y = 1) )$ and $( p(x_{\text{test}} \mid y = 2) )$  using the multivariate normal distribution formula.
2. **Use Bayes' theorem to calculate the posterior probabilities $( p(y = 1 \mid x_{\text{test}}) )$ and $( p(y = 2 \mid x_{\text{test}}) )$.
3. **Compare the posterior probabilities to classify the new test point.**

### Detailed Calculation:

1. **Multivariate Normal Distribution Formula:**

$$ p(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{(2 \pi)^{k/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right) $$

where \( k \) is the number of dimensions (in this case, 2).

2. **Class-Conditional Probabilities:**

**For \( y = 1 \):**

$$ \mu^{(1)} = \begin{bmatrix} 0.77 \\ -0.41 \end{bmatrix}, \quad \Sigma^{(1)} = \begin{bmatrix} 0.29 & -0.12 \\ -0.12 & 0.55 \end{bmatrix} $$

$$ \Sigma^{(1)^{-1}} = \begin{bmatrix} \frac{0.55}{(0.29 \cdot 0.55 - (-0.12)^2)} & \frac{0.12}{(0.29 \cdot 0.55 - (-0.12)^2)} \\ \frac{0.12}{(0.29 \cdot 0.55 - (-0.12)^2)} & \frac{0.29}{(0.29 \cdot 0.55 - (-0.12)^2)} \end{bmatrix} \approx \begin{bmatrix} 3.92 & 0.85 \\ 0.85 & 2.07 \end{bmatrix} $$

$$ |\Sigma^{(1)}| = 0.29 \cdot 0.55 - (-0.12)^2 = 0.1163 $$

$$ p(x_{\text{test}} \mid y = 1) = \frac{1}{2 \pi \sqrt{0.1163}} \exp \left( -\frac{1}{2} \left( \begin{bmatrix} 0 \\ 0.7 \end{bmatrix} - \begin{bmatrix} 0.77 \\ -0.41 \end{bmatrix} \right)^T \begin{bmatrix} 3.92 & 0.85 \\ 0.85 & 2.07 \end{bmatrix} \left( \begin{bmatrix} 0 \\ 0.7 \end{bmatrix} - \begin{bmatrix} 0.77 \\ -0.41 \end{bmatrix} \right) \right) $$

$$ = \frac{1}{2 \pi \sqrt{0.1163}} \exp \left( -\frac{1}{2} \begin{bmatrix} -0.77 \\ 1.11 \end{bmatrix}^T \begin{bmatrix} 3.92 & 0.85 \\ 0.85 & 2.07 \end{bmatrix} \begin{bmatrix} -0.77 \\ 1.11 \end{bmatrix} \right) $$

$$ \approx \frac{1}{2 \pi \sqrt{0.1163}} \exp \left( -\frac{1}{2} (2.3212 + 0.9182 + 0.9182 + 2.5581) \right) $$

$$ \approx \frac{1}{2 \pi \sqrt{0.1163}} \exp \left( -3.3575 \right) $$

$$ \approx \frac{1}{0.1082} \exp \left( -3.3575 \right) \approx 9.24 \times 10^{-3} $$

**For \( y = 2 \):**

$$ \mu^{(2)} = \begin{bmatrix} -0.91 \\ 0.5 \end{bmatrix}, \quad \Sigma^{(2)} = \begin{bmatrix} 0.32 & -0.11 \\ -0.11 & 1.12 \end{bmatrix} $$

$$ \Sigma^{(2)^{-1}} = \begin{bmatrix} \frac{1.12}{(0.32 \cdot 1.12 - (-0.11)^2)} & \frac{0.11}{(0.32 \cdot 1.12 - (-0.11)^2)} \\ \frac{0.11}{(0.32 \cdot 1.12 - (-0.11)^2)} & \frac{0.32}{(0.32 \cdot 1.12 - (-0.11)^2)} \end{bmatrix} \approx \begin{bmatrix} 3.34 & 0.33 \\ 0.33 & 0.95 \end{bmatrix} $$

$$ |\Sigma^{(2)}| = 0.32 \cdot 1.12 - (-0.11)^2 = 0.3456 $$

$$ p(x_{\text{test}} \mid y = 2) = \frac{1}{2 \pi \sqrt{0.3456}} \exp \left( -\frac{1}{2} \left( \begin{bmatrix} 0 \\ 0.7 \end{bmatrix} - \begin{bmatrix} -0.91 \\ 0.5 \end{bmatrix} \right)^T \begin{bmatrix} 3.34 & 0.33 \\ 0.33 & 0.95 \end{bmatrix} \left( \begin{bmatrix} 0 \\ 0.7 \end{bmatrix} - \begin{bmatrix} -0.91 \\ 0.5 \end{bmatrix} \right) \right) $$

$$ = \frac{1}{2 \pi \sqrt{0.3456}} \exp \left( -\frac{1}{2} \left( \begin{bmatrix} 0.91 \\ 0.2 \end{bmatrix} \right)^T \begin{bmatrix} 3.34 & 0.33 \\ 0.33 & 0.95 \end{bmatrix} \begin{bmatrix} 0.91 \\ 0.2 \end{bmatrix} \right) $$

$$ \approx \frac{1}{2 \pi \sqrt{0.3456}} \exp \left( -\frac{1}{2} (2.7634 + 0.6006 + 0.066 + 0.095) \right) $$

$$ \approx \frac{1}{2 \pi \sqrt{0.3456}} \exp \left( -1.775 \right) $$

$$ \approx \frac{1}{0.296} \exp \left( -1.775 \right) \approx 0.1805 $$

3. **Calculate Posterior Probabilities:**

$$ p(y = 1 \mid x_{\text{test}}) = \frac{p(x_{\text{test}} \mid y = 1)

 p(y = 1)}{p(x_{\text{test}})} $$

$$ p(y = 2 \mid x_{\text{test}}) = \frac{p(x_{\text{test}} \mid y = 2) p(y = 2)}{p(x_{\text{test}})} $$

Since $( p(x_{\text{test}}))$ is a common denominator, we compare the numerators:

$$ p(y = 1 \mid x_{\text{test}}) \propto p(x_{\text{test}} \mid y = 1) p(y = 1) \approx 9.24 \times 10^{-3} \times 0.53 \approx 4.8972 \times 10^{-3} $$

$$ p(y = 2 \mid x_{\text{test}}) \propto p(x_{\text{test}} \mid y = 2) p(y = 2) \approx 0.1805 \times 0.47 \approx 0.084835 $$

### Conclusion:

Since $( p(y = 2 \mid x_{\text{test}}) )$ is much larger than $( p(y = 1 \mid x_{\text{test}}) )$, the classifier will classify the point to the High class $( y = 2)$.

The correct answer is:
**C. We will classify the point to the High class (y = 2).**


### Script
```
import numpy as np # type: ignore  
from scipy.stats import norm # type: ignore  
  
# Given data  
mu1 = np.array([0.77, -0.41])  
sigma1_diag = np.array([0.29, 0.55])  
  
mu2 = np.array([-0.91, 0.5])  
sigma2_diag = np.array([0.32, 1.12])  
  
x_test = np.array([0, 0.7])  
  
# Priors  
p_y1 = 0.53  
p_y2 = 0.47  
  
# Likelihoods using only diagonal elements  
likelihood_y1 = (norm.pdf(x_test[0], mu1[0], np.sqrt(sigma1_diag[0])) *  
                 norm.pdf(x_test[1], mu1[1], np.sqrt(sigma1_diag[1])))  
  
likelihood_y2 = (norm.pdf(x_test[0], mu2[0], np.sqrt(sigma2_diag[0])) *  
                 norm.pdf(x_test[1], mu2[1], np.sqrt(sigma2_diag[1])))  
  
# Posteriors  
posterior_y1 = likelihood_y1 * p_y1  
posterior_y2 = likelihood_y2 * p_y2  
  
print(f"Posterior probability for y=1: {posterior_y1:.6f}")  
print(f"Posterior probability for y=2: {posterior_y2:.6f}")  
  
# Classify based on higher posterior probability  
if posterior_y1 > posterior_y2:  
    classification = "y=1 (Low class)"  
else:  
    classification = "y=2 (High class)"  
  
print(classification)
```
---
## F16, Q14) What is the probability that a person that has occurrence of nausea, i.e. x1 = 1, has inflammation of the urinary bladder, i.e. y = 1, according to this study?
![[Pasted image 20240521151227.png]]

---
## F16, Q17) Given that a person has x1 = 1, and x2 = 1 what is the probability that the person has an inflammation of urinary bladder (y = 1) according to the Na ̈ıve Bayes classifier?

![[Pasted image 20240521151538.png]]

![[Pasted image 20240521151458.png]]

---
## F17, Q14) 

![[Pasted image 20240521155654.png]]

---
## S20, Q16)

![[Pasted image 20240521200828.png]]
![[Pasted image 20240521200821.png]]


```python 

import numpy as np  
  
  
def calculate_probability(table, alpha=1):  
    # Convert the table to a numpy array for easy manipulation  
    data = np.array(table)  
  
    # Extract the relevant columns  
    f2 = data[:, 1]  
    f3 = data[:, 2]  
    yb = data[:, -1]  
  
    # Calculate occurrences  
    occurrences_y1 = np.sum(yb == 1)  
    occurrences_f2_1_f3_1_y1 = np.sum((f2 == 1) & (f3 == 1) & (yb == 1))  
  
    # Calculate the probability using the given formula  
    probability = (occurrences_f2_1_f3_1_y1 + alpha) / (occurrences_y1 + 2 * alpha)  
  
    return probability  
  
  
# Example table data: Each row is [f1, f2, f3, f4, f5, yb]
# yb is determined by the class color black = 1 and red = 0 in this case
table = [  
    [1, 1, 1, 0, 0, 0],  
    [1, 1, 1, 0, 0, 0],  
    [1, 1, 1, 0, 0, 0],  
    [1, 1, 1, 0, 0, 0],  
    [1, 1, 1, 0, 0, 0],  
    [0, 1, 1, 0, 0, 0],  
    [0, 1, 0, 1, 1, 0],  
    [1, 1, 1, 0, 0, 0],  
    [1, 0, 1, 0, 0, 1],  
    [0, 0, 0, 1, 1, 1],  
    [0, 1, 0, 1, 1, 1]  
]  
  
# Calculate the probability  
prob = calculate_probability(table, alpha=1)  
print(f"The estimated probability is: {prob}")

```
