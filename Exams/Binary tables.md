
## S19, Q14) Which of the following options represents all (non-empty) itemsets with support greater than 0.15 (and only itemsets with support greater than 0.15)?

![[Pasted image 20240516001614.png]]


```python
import itertools  
  
  
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
