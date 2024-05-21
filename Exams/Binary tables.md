
## ?S19, Q14) Which of the following options represents all (non-empty) itemsets with support greater than 0.15 (and only itemsets with support greater than 0.15)? 

![[Pasted image 20240516001614.png]]


```python
import itertools 
import numpy as np
  
  
def find_extended_itemsets_with_support(table, num_items, min_support_ratio):  
    """  
    Find all itemsets (single items, pairs, triples, etc.) that occur with at least a given support ratio.  
    :param table: List of transactions, where each transaction is represented by a list of item flags (0 or 1)    :param num_items: The number of items (up to 10 in this case)    :param min_support_ratio: The minimum support ratio    :return: A list of itemsets that meet the minimum support ratio    """    itemsets = []  
    num_transactions = len(table)  
    min_support_count = min_support_ratio * num_transactions  
  
    # Generate all possible itemsets of size 1 to min(num_items, 10)  
    for size in range(1, min(num_items, 10) + 1):  
        for itemset in itertools.combinations(range(num_items), size):  
            support_count = sum(all(transaction[i] for i in itemset) for transaction in table)  
            if support_count >= min_support_count:  
                itemsets.append({f'f{i+1}' for i in itemset})  
  
    return itemsets  
  
# Sample table of transactions  
table_data = [  
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Transaction 1  
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Transaction 2  
    [0, 1, 1, 1, 1, 1, 0, 0, 0],  # Transaction 3  
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Transaction 4  
    [1, 0, 0, 1, 0, 0, 0, 0, 0],  # Transaction 5  
    [0, 0, 1, 1, 0, 0, 0, 1, 0],  # Transaction 6  
    [0, 0, 1, 1, 1, 0, 0, 0, 0],  # Transaction 7  
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Transaction 8  
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # Transaction 9  
    [0, 0, 1, 1, 1, 0, 1, 0, 0]  # Transaction 10  
]  
  
# Set the number of items and the minimum support ratio  
num_items = 9  # Number of items  
min_support_ratio = 0.15  # Minimum support ratio  
  
# Calculate extended itemsets with support greater than the given ratio  
extended_itemsets_with_support = find_extended_itemsets_with_support(table_data, num_items, min_support_ratio)  
print("Extended itemsets with sufficient support:", extended_itemsets_with_support)
```

## S19, Q15)  What is the confidence of the rule {f2} → {f3, f4, f5, f6}?

![[Pasted image 20240516001740.png]]

![[Pasted image 20240516002858.png]]

Support = times they appear in table, note 1/5 = 2/10


## S19, Q16) What, in the notation of the lecture notes, is C2? #Apriori

![[Pasted image 20240516004236.png]]


### Understanding A-priori Algorithm:

The **A-priori Algorithm** is a popular method used in data mining to identify frequent itemsets (combinations of items) and derive association rules from a database of transactions. Each item in a transaction is treated as a binary feature, either present (1) or absent (0).

### Key Concepts:

1. **Support**: This measures how frequently an itemset appears in the dataset. An itemset must meet a minimum support threshold to be considered frequent.

2. **\(L1\) Itemsets**: These are single items that meet the minimum support threshold on the first pass of the algorithm.

3. **\(C2\) Itemsets**: These are combinations of two items generated from \(L1\) itemsets that you need to check in the next round to see if they also meet the support threshold.

### How A-priori Works:

- **Step 1**: Determine the support of each individual item and keep only those that meet a minimum support threshold. These make up the \(L1\) itemsets.

- **Step 2**: Combine \(L1\) itemsets to form \(C2\) itemsets, which are candidate itemsets containing two items.

- **Step 3**: Calculate the support for each candidate itemset in \(C2\) and keep those that meet the threshold, forming \(L2\) itemsets.

### Your Specific Example:

From your information, you have:
- \(L1\) includes itemsets \([0, 0, 1, 0]\) and \([0, 0, 0, 1]\). This means the single items \(f3\) and \(f4\) have sufficient support on their own.

### Generating \(C2\):

- You take each item in \(L1\) and pair them with each other to form \(C2\). Since there are only two items in \(L1\), the only possible combination is \(f3\) and \(f4\) together:

  - \(f3\) is represented as \([0, 0, 1, 0]\)
  - \(f4\) is represented as \([0, 0, 0, 1]\)

- When combined:

  - The binary representation of \(f3\) and \(f4\) together is \([0, 0, 1, 1]\), which means both \(f3\) and \(f4\) are present in the itemset.

## S19, Q16) Consider the observations in Table 4. We consider these as 9-dimensional binary vectors and wish to compute the pairwise similarity. Which of the following statements are true? #jaccard #cosine #smc

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
  
  
# Observations from Table 4 as 5-dimensional binary vectors  
o1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])  
o2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  
o3 = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0])  
o4 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])  
  
# Calculate similarities and coefficients  
cos_o1_o3 = cosine_similarity(o1, o3)  
smc_o1_o3 = simple_matching_coefficient(o1, o3)  
smc_o2_o4 = simple_matching_coefficient(o3, o4)  
j_o2_o3 = jaccard_similarity(o2, o3)  
  
# Print the results  
print("Cosine Similarity between o1 and o3:", cos_o1_o3)  
print("Jaccard Similarity between o2 and o3:", j_o2_o3)  
print("Simple Matching Coefficient between o1 and o3:", smc_o1_o3)  
print("Simple Matching Coefficient between o2 and o4:", smc_o2_o4)  
  
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
## F23, Q8) What is the minimum and maximum achievable value of the Simple Matching Coefficient (SMC) between o6 from Table 2 and the new observation o11? #smc 

![[Pasted image 20240517235352.png]]


Consider a new, binarized observation $( \mathbf{o}_{11} = [0, ?, ?, 1]^T )$, which contains two unknown values each indicated with a ´?´. What is the minimum and maximum achievable value of the Simple Matching Coefficient (SMC) between $( \mathbf{o}_{6})$ from Table 2 and the new observation $( \mathbf{o}_{11} )$?

A. 
$$
\text{min SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.50 \\
\text{max SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 1.0
$$

B. 
$$
\text{min SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.75 \\
\text{max SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.9
$$

C. 
$$
\text{min SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.25 \\
\text{max SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 1.0
$$

D. 
$$
\text{min SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.25 \\
\text{max SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.9
$$

E. Don’t know.

#### Explanation:
To solve this question, let's break down the steps needed to calculate the Simple Matching Coefficient (SMC) and determine the minimum and maximum values it can take.

1. **Simple Matching Coefficient (SMC):**
   - The SMC between two binary vectors is defined as the number of matching components (both 0s and 1s) divided by the total number of components.
   - Mathematically:
     $$
     \text{SMC} = \frac{(a + d)}{(a + b + c + d)}
     $$
     where:
     - \( a \) is the number of times both components are 0.
     - \( d \) is the number of times both components are 1.
     - \( b \) is the number of times the first component is 0 and the second is 1.
     - \( c \) is the number of times the first component is 1 and the second is 0.

2. **Given Information:**
   - Observation $( \mathbf{o}_{6})$ from Table 2: $( [0, 1, 0, 1] )$
   - New observation $( \mathbf{o}_{11} = [0, ?, ?, 1]^T)$ with unknown values represented by ?.

3. **Match with Known Components:**
   - The 1st component of $( \mathbf{o}_{6})$ and $( \mathbf{o}_{11})$ both are 0 (matches).
   - The 4th component of $( \mathbf{o}_{6})$ and $( \mathbf{o}_{11})$ both are 1 (matches).

4. **Determine Possible Values for Unknown Components:**
   - Let the unknown components be $( \mathbf{o}_{11,2})$ and $( \mathbf{o}_{11,3})$.

5. **Calculate Minimum and Maximum SMC:**
   - **Maximum SMC:**
     - To maximize SMC, we want the unknown components to match the corresponding components of $( \mathbf{o}_{6})$.
     - So, set $( \mathbf{o}_{11,2} = 1)$ and $( \mathbf{o}_{11,3} = 0)$.
     - Matching components: 4 (all components match).
     - SMC = $( \frac{4}{4} = 1.0)$.

   - **Minimum SMC:**
     - To minimize SMC, we want the unknown components to not match the corresponding components of $( \mathbf{o}_{6})$.
     - So, set $( \mathbf{o}_{11,2} = 0)$ and $( \mathbf{o}_{11,3} = 1 )$.
     - Matching components: 2 (1st and 4th components match).
     - Non-matching components: 2 (2nd and 3rd components do not match).
     - SMC = $( \frac{2}{4} = 0.5)$.

### Conclusion:
The correct answer is:
A. 
$$
\text{min SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 0.50 \\
\text{max SMC}(\mathbf{o}_{6}, \mathbf{o}_{11}) = 1.0
$$

### Summary Notes:
1. **Simple Matching Coefficient (SMC):**
   - Measures similarity between two binary vectors.
   - Formula: $( \text{SMC} = \frac{(a + d)}{(a + b + c + d)})$.

2. **Calculation Strategy:**
   - Identify matching and non-matching components.
   - Consider both maximum and minimum scenarios for unknown components.

3. **Key Steps:**
   - Set unknown components to maximize and minimize matches.
   - Compute SMC for both scenarios to find the range.

### Script
```python
import numpy as np  
  
def compute_smc(o1, o2):  
    """  
    Compute the Simple Matching Coefficient (SMC) between two binary vectors.    Parameters:    o1 (list or np.ndarray): First binary vector    o2 (list or np.ndarray): Second binary vector    Returns:    float: Simple Matching Coefficient    """    o1 = np.array(o1)  
    o2 = np.array(o2)  
    matches = np.sum(o1 == o2)  
    total = len(o1)  
    smc = matches / total  
    return smc  
  
def calculate_min_max_smc(o6, o11):  
    """  
    Calculate the minimum and maximum SMC between o6 and o11 with unknown values in o11.    Parameters:    o6 (list): Binary vector from table    o11 (list): Binary vector with unknown values represented by None    Returns:    tuple: (min_smc, max_smc)    """    # Known parts of o11  
    known_indices = [i for i, x in enumerate(o11) if x is not None]  
    unknown_indices = [i for i, x in enumerate(o11) if x is None]  
  
    # Known matches  
    known_matches = sum(o6[i] == o11[i] for i in known_indices)  
  
    # Minimum SMC: Assume all unknown values do not match  
    min_matches = known_matches  
    min_smc = min_matches / len(o6)  
  
    # Maximum SMC: Assume all unknown values match  
    max_matches = known_matches + len(unknown_indices)  
    max_smc = max_matches / len(o6)  
  
    return min_smc, max_smc  
  
# Observation o6 from Table 2  
o6 = [0, 1, 0, 1]  
  
# New observation o11 with unknown values represented by None  
o11 = [0, None, None, 1]  
  
# Calculate minimum and maximum SMC  
min_smc, max_smc = calculate_min_max_smc(o6, o11)  
  
print(f"Minimum SMC(o6, o11): {min_smc:.2f}")  
print(f"Maximum SMC(o6, o11): {max_smc:.2f}")  
  
# Based on the given options, the correct answer is:  
# A.  
# min SMC(o6, o11) = 0.50  
# max SMC(o6, o11) = 1.0
```

---
## F17, Q21) What color will o10 be classified as #jaccard 
![[Pasted image 20240521160512.png]]

![[Pasted image 20240521160515.png]]

```python
from scipy.spatial.distance import jaccard # type: ignore  
  
# Hypothetical binary data for O1 to O10  
#TODO: update based on dataset  
data = [  
    [1, 0, 1, 0, 1, 0, 1, 0],  
    [1, 0, 1, 0, 1, 0, 1, 0],  
    [1, 0, 1, 0, 1, 0, 1, 0],  
    [1, 0, 1, 0, 1, 0, 0, 1],  
    [1, 0, 1, 0, 0, 1, 0, 1],  
    [1, 0, 0, 1, 0, 1, 1, 0],  
    [0, 1, 1, 0, 0, 1, 0, 1],  
    [0, 1, 1, 0, 1, 0, 0, 1],  
    [0, 1, 0, 1, 1, 0, 1, 0],  
    [0, 1, 0, 1, 0, 1, 1, 0]  
]  
  
# Compute Jaccard distance (1 - Jaccard similarity index) for O10 against others  
#TODO: Change data[9] based on which observation it has to be for  
o10 = data[9]  
  
jaccard_scores = [('O' + str(i+1), 1 - jaccard(o10, data[i])) for i in range(9)]  
  
# Sort based on highest similarity (lowest distance, since we used 1 - similarity)  
sorted_scores = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)  
  
# Get top 3 similar observations  
top_three = sorted_scores[:3]  
  
# Display results  
print("We can see the 3 nearest neighbor classifier to classify observation is")  
print(top_three)
```
---
## S23, Q24) What is the importance of the classifier f1? #AdaBoost 

![[Pasted image 20240521031659.png]]
## Script

```python
import numpy as np  
  
# Binarized dataset (Table 8)  
data = np.array([  
    [1, 1, 0, 1, 0, 0, 'C1'],  
    [1, 0, 1, 1, 0, 0, 'C1'],  
    [0, 1, 1, 1, 1, 1, 'C1'],  
    [0, 1, 1, 1, 1, 1, 'C1'],  
    [1, 1, 0, 1, 1, 0, 'C2'],  
    [0, 1, 1, 0, 1, 1, 'C2'],  
    [0, 1, 1, 1, 0, 1, 'C2'],  
    [0, 1, 0, 1, 1, 1, 'C2'],  
    [0, 1, 0, 1, 1, 1, 'C2'],  
    [1, 1, 1, 1, 1, 0, 'C2']  
])  
  
# Classifier f1: f1(b1, b2, b3, b4, b5, b6) = C1 if b3 = 1 and b4 = 1, otherwise C2  
#TODO update based on what the task says  
def classifier_f1(b):  
    b3, b4 = b[2], b[3]  
    if b3 == 1 and b4 == 1:  
        return 'C1'  
    else:  
        return 'C2'  
  
# Calculate predictions and actual classes  
predictions = []  
actual_classes = data[:, -1]  
for i in range(data.shape[0]):  
    b = data[i, :-1].astype(int)  
    prediction = classifier_f1(b)  
    predictions.append(prediction)  
  
# Calculate error rate epsilon_1  
misclassified = np.sum(predictions != actual_classes)  
total = data.shape[0]  
epsilon_1 = misclassified / total  
  
# Calculate the importance alpha_1 of the classifier f1  
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)  
  
# Print the results  
print(f"Predictions: {predictions}")  
print(f"Actual Classes: {actual_classes.tolist()}")  
print(f"Misclassified: {misclassified}")  
print(f"Total Observations: {total}")  
print(f"Error Rate (epsilon_1): {epsilon_1:.2f}")  
print(f"Importance (alpha_1): {alpha_1:.2f}")
```
