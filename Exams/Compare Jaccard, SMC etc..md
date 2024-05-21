
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