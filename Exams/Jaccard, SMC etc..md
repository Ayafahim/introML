
## F16, Q19) #jaccard  #smc  #pnormdistance #cosine 

```python
import numpy as np  # type: ignore  
  
  
# Examset 16 spring, task 13  
def cosine_similarity(vec1, vec2):  
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  
  
  
def simple_matching_coefficient(vec1, vec2):  
    return sum(1 for i, j in zip(vec1, vec2) if i == j) / len(vec1)  
  
  
def jaccard_similarity(vec1, vec2):  
    intersection = sum(1 for i, j in zip(vec1, vec2) if i == j and i == 1)  
    union = sum(1 for i, j in zip(vec1, vec2) if i == 1 or j == 1)  
    return intersection / union  
  
  
def p_norm_distance(vec1, vec2, p):  
    return np.linalg.norm(vec1 - vec2, ord=p)  
  
  
# Observations from Table 4 as 5-dimensional binary vectors  
o1 = np.array([1, 0, 1, 0, 0, 1])  
o2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  
o3 = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0])  
o4 = np.array([1, 0, 1, 0, 1, 0])  
  
x35 = np.array([-1.24, -0.26, -1.04])  
x53 = np.array([-0.60, -0.86, -0.50])  
  
r = [1, 1, 1, 1, 0, 1]  
s = [1, 1, 0, 1, 0, 0]  
  
p_norm2 = p_norm_distance(o1, o4, 2)  
p_norm1 = p_norm_distance(o1, o4, 1)  
j = jaccard_similarity(o1, o4)  
cos = cosine_similarity(o1, o4)  
smc = simple_matching_coefficient(o1, o4)  
  
print("p_norm2", p_norm2)  
print("p_norm1", p_norm1)  
print("j", j)  
print("cos", cos)  
print("smc", smc)  
  
# Calculate similarities and coefficients  
# cos_o1_o3 = cosine_similarity(o1, o3)  
# smc_o1_o3 = simple_matching_coefficient(o1, o3)  
# smc_o2_o4 = simple_matching_coefficient(o3, o4)  
# j_o2_o3 = jaccard_similarity(o2, o3)  
  
###  
# xcos = cosine_similarity(x35, x53)  
# x_pnorm1 = p_norm_distance(x35, x53, 1)  
# x_pnorm4 = p_norm_distance(x35, x53, 4)  
# x_pnormInf = p_norm_distance(x35, x53, np.inf)  
  
###  
# j = jaccard_similarity(r, s)  
# smc = simple_matching_coefficient(r, s)  
# c = cosine_similarity(r, s)  
  
  
# Print the results  
# print("Cosine Similarity between o1 and o3:", cos_o1_o3)  
# print("Jaccard Similarity between o2 and o3:", j_o2_o3)  
# print("Simple Matching Coefficient between o1 and o3:", smc_o1_o3)  
# print("Simple Matching Coefficient between o2 and o4:", smc_o2_o4)  
# print(xcos)  
# print("1", x_pnorm1)  
# print("4", x_pnorm4)  
# print("inf", x_pnormInf)  
#  
# print("Jaccard Similarity between r and s:", j)  
# print("Simple Matching Coefficient between r and s:", smc)  
# print("Cosine Similarity between r and s:", cosine_similarity(r, s))  
#  
statement_a = p_norm2 == 2  
statement_b = p_norm1 < p_norm2  
statement_c = j == 2/3  
statement_d = cos == smc  
  
print("\nValidity of the statements:")  
print("Statement A", statement_a)  
print("Statement B", statement_b)  
print("Statement C", statement_c)  
print("Statement D", statement_d)  
  
# # Check the validity of the statements  
# statement_a = cos_o1_o3 > smc_o1_o2  
# statement_b = cos_o1_o2 > cos_o1_o3  
# statement_c = j_o1_o3 > smc_o1_o2  
# statement_d = j_o1_o3 > cos_o1_o3  
#  
# print("\nValidity of the statements:")  
# print("Statement A", statement_a)  
# print("Statement B", statement_b)  
# print("Statement C", statement_c)  
# print("Statement D", statement_d)
```


---

## S22, Q9)What are the possible range of values of the Jaccard similarities of $x_1$  and $x_2$ ?
![[Pasted image 20240521215524.png]]


To solve Question 9 regarding the range of possible values of the Jaccard similarities for two binary vectors $x_1$ and $x_2$ with dimensions $N = 465$, where $x_1$ has one non-zero element and $x_2$ has 464 non-zero elements, we need to understand how the Jaccard similarity is calculated.

The Jaccard similarity between two binary vectors $x_1$ and $x_2$ is defined as:

$$ J(x_1, x_2) = \frac{|x_1 \cap x_2|}{|x_1 \cup x_2|} $$

Where:
- $|x_1 \cap x_2|$ is the number of positions where both vectors have a 1.
- $|x_1 \cup x_2|$ is the number of positions where at least one of the vectors has a 1.

Given the vectors:
- $x_1$ has one non-zero element.
- $x_2$ has 464 non-zero elements.

### Calculation:

1. **Intersection $|x_1 \cap x_2|$**:
   - The maximum possible value for the intersection is 1 (if the one non-zero element of $x_1$ is one of the 464 non-zero elements of $x_2$).
   - The minimum possible value for the intersection is 0 (if the one non-zero element of $x_1$ is not among the 464 non-zero elements of $x_2$).

2. **Union $|x_1 \cup x_2|$**:
   - Since $x_1$ has one non-zero element and $x_2$ has 464 non-zero elements, the total number of unique non-zero positions in the union is either 464 or 465:
     - If the one non-zero element of $x_1$ is one of the 464 non-zero elements of $x_2$, then $|x_1 \cup x_2| = 464$.
     - If the one non-zero element of $x_1$ is not among the 464 non-zero elements of $x_2$, then $|x_1 \cup x_2| = 465$.

### Jaccard Similarity Range:

- **Maximum Jaccard Similarity**:
  $$ J_{\text{max}} = \frac{1}{464} \approx 0.00216 $$

- **Minimum Jaccard Similarity**:
  $$ J_{\text{min}} = \frac{0}{465} = 0 $$

Thus, the range of the Jaccard similarity values is:

$$ J(x_1, x_2) \in [0, 0.00216] $$

The correct answer is:

$$ \boxed{C. J(x_1, x_2) \in [0; 0.00216]} $$

```python
def jaccard_similarity(x1, x2):
    """
    Calculate the Jaccard similarity between two binary vectors x1 and x2.
    """
    intersection = sum([1 for i, j in zip(x1, x2) if i == j == 1])
    union = sum([1 for i, j in zip(x1, x2) if i == 1 or j == 1])
    return intersection / union

# Define the binary vectors
N = 465
x1 = [0] * N
x2 = [1] * N

# Place one non-zero element in x1
x1[0] = 1

# Calculate Jaccard similarity when the non-zero element in x1 is also in x2 (maximum intersection)
jaccard_max = jaccard_similarity(x1, x2)

# Calculate Jaccard similarity when the non-zero element in x1 is not in x2 (minimum intersection)
x2[0] = 0  # Move the non-zero element of x1 out of the 464 non-zero elements of x2
jaccard_min = jaccard_similarity(x1, x2)

print(f"Jaccard similarity range: [{jaccard_min:.5f}, {jaccard_max:.5f}]")

```
