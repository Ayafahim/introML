
## S19, Q7) Jaccard similarity of clusters

![[Pasted image 20240515173540.png]]

With the cut off you can see the clusters are where the line stops connecting. So $Q=\{o_1,o_4,o_2,o_5,o_6,o_7\},\{o_{10}\},\{o_3,o_8,o_9\}$

#### Cluster $C_1$ in $\mathbb{Z}$ vs. Clusters in $\mathbb{Q}$:

$C_1$ (Black) in $\mathbb{Z}$: 
- $\{1, 2\}$ $\{o_1, o_2\}$

In $\mathbb{Q}$:
- Cluster 1: $\{10\}$ $\{o_{10}\}$ – No overlap
- Cluster 2: $\{1, 2, 4, 5, 6, 7\}$ $\{o_1, o_2, o_4, o_5, o_6, o_7\}$ – Contains $1o_1$ and $2o_2$ (2 points)
- Cluster 3: $\{3, 8, 9\}$ $\{o_3, o_8, o_9\}$ – No overlap

So, the first row of $n$ is $[0, 2, 0]$.

#### Cluster $C_2$ in $\mathbb{Z}$ vs. Clusters in $\mathbb{Q}$:

$C_2$ (Red) in $\mathbb{Z}$: 
- $\{3, 4, 5\}$ $\{o_3, o_4, o_5\}$

In $\mathbb{Q}$:
- Cluster 1: $\{10\}$ $\{o_{10}\}$ – No overlap
- Cluster 2: $\{1, 2, 4, 5, 6, 7\}$ $\{o_1, o_2, o_4, o_5, o_6, o_7\}$ – Contains $4o_4$ and $5o_5$ (2 points)
- Cluster 3: $\{3, 8, 9\}$ $\{o_3, o_8, o_9\}$ – Contains $3o_3$ (1 point)

So, the second row of $n$ is $[0, 2, 1]$.

#### Cluster $C_3$ in $\mathbb{Z}$ vs. Clusters in $\mathbb{Q}$:

$C_3$ (Blue) in $\mathbb{Z}$: 
- $\{6, 7, 8, 9, 10\}$ $\{o_6, o_7, o_8, o_9, o_{10}\}$

In $\mathbb{Q}$:
- Cluster 1: $\{10\}$ $\{o_{10}\}$ – Contains $10o_{10}$ (1 point)
- Cluster 2: $\{1, 2, 4, 5, 6, 7\}$ $\{o_1, o_2, o_4, o_5, o_6, o_7\}$ – Contains $6o_6$ and $7o_7$ (2 points)
- Cluster 3: $\{3, 8, 9\}$ $\{o_3, o_8, o_9\}$ – Contains $8o_8$ and $9o_9$ (2 points)

So, the third row of $n$ is $[1, 2, 2]$.


so we get counting matrix n:
$$
n = \begin{bmatrix}
0 & 2 & 0 \\
0 & 2 & 1 \\
1 & 2 & 2
\end{bmatrix}
$$
Then just input the matrix into the script and calculate  S, D and J similarity:

```python
import numpy as np

def calculate_jaccard(matrix):
    n = np.array(matrix)
    N = np.sum(n)  # Total number of elements, assuming matrix entries are counts
    S = np.sum([n[k,m] * (n[k,m] - 1) / 2 for k in range(n.shape[0]) for m in range(n.shape[1])])

    nZ = np.sum(n, axis=1)  # Sum of rows
    nQ = np.sum(n, axis=0)  # Sum of columns

    sum_nZ = np.sum([nZ[k] * (nZ[k] - 1) / 2 for k in range(len(nZ))])
    sum_nQ = np.sum([nQ[m] * (nQ[m] - 1) / 2 for m in range(len(nQ))])

    D = N * (N - 1) / 2 - sum_nZ - sum_nQ + S
    J = S / (0.5 * N * (N - 1) - D)  # Jaccard similarity formula

    return S, D, J

# Example matrix
n = [
    [0, 2, 0],
    [0, 2, 1],
    [1, 2, 2]
]

S, D, J = calculate_jaccard(n)
print("S:", S, "D:", D, "Jaccard Similarity:", J)
```



---

## S23, Q7) What is the Rand index of the two clusterings? 
![[Pasted image 20240520202501.png]]

### Solution 

Just use this script, change it for whatever type of index you need.

```python
import numpy as np  
from scipy.special import comb  
  
  
import numpy as np
from scipy.special import comb


def create_contingency_matrix(Z, Q):
    """
    Create a contingency matrix from clusters Z and Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    np.ndarray: Contingency matrix
    """
    contingency_matrix = np.zeros((len(Z), len(Q)), dtype=int)
    for i, c in enumerate(Z):
        for j, q in enumerate(Q):
            contingency_matrix[i, j] = len(c & q)  # Intersection of sets
    return contingency_matrix


def compute_pair_counts(contingency_matrix):
    """
    Compute the pair counts (a, b, c, d) from the contingency matrix.
    Parameters:
    contingency_matrix (np.ndarray): Contingency matrix
    Returns:
    tuple: (a, b, c, d) pair counts
    """
    # Number of elements
    N = np.sum(contingency_matrix)

    # Sum of combinations of pairs within each cell
    sum_comb = np.sum(comb(contingency_matrix, 2))

    # Sum of combinations of pairs within each row
    sum_comb_rows = np.sum(comb(np.sum(contingency_matrix, axis=1), 2))

    # Sum of combinations of pairs within each column
    sum_comb_cols = np.sum(comb(np.sum(contingency_matrix, axis=0), 2))

    # Total number of pairs
    total_pairs = comb(N, 2)

    # Number of agreements (S)
    a = sum_comb

    # Number of disagreements
    b = sum_comb_rows - a
    c = sum_comb_cols - a
    d = total_pairs - sum_comb_rows - sum_comb_cols + a

    return a, b, c, d


def compute_rand_index(Z, Q):
    """
    Compute the Rand Index from ground truth clusters Z and predicted clusters Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    float: Rand Index
    """
    contingency_matrix = create_contingency_matrix(Z, Q)
    a, b, c, d = compute_pair_counts(contingency_matrix)
    total_pairs = comb(np.sum(contingency_matrix), 2)
    rand_index = (a + d) / total_pairs
    return rand_index


def compute_jaccard_index(Z, Q):
    """
    Compute the Jaccard Index from ground truth clusters Z and predicted clusters Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    float: Jaccard Index
    """
    contingency_matrix = create_contingency_matrix(Z, Q)
    a, b, c, d = compute_pair_counts(contingency_matrix)
    jaccard_index = a / (a + b + c)
    return jaccard_index


def compute_smc(Z, Q):
    """
    Compute the Simple Matching Coefficient (SMC) from ground truth clusters Z and predicted clusters Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    float: Simple Matching Coefficient
    """
    contingency_matrix = create_contingency_matrix(Z, Q)
    a, b, c, d = compute_pair_counts(contingency_matrix)
    total_pairs = comb(np.sum(contingency_matrix), 2)
    smc = (a + d) / total_pairs
    return smc


def compute_entropy(prob):
    """
    Compute the entropy given a probability distribution.
    Parameters:
    prob (np.ndarray): Probability distribution
    Returns:
    float: Entropy
    """
    return -np.sum(prob * np.log(prob + 1e-12))


def compute_mutual_information(contingency_matrix):
    """
    Compute the mutual information from the contingency matrix.
    Parameters:
    contingency_matrix (np.ndarray): Contingency matrix
    Returns:
    float: Mutual information
    """
    total = np.sum(contingency_matrix)
    prob_matrix = contingency_matrix / total
    prob_Z = np.sum(prob_matrix, axis=1)
    prob_Q = np.sum(prob_matrix, axis=0)
    entropy_Z = compute_entropy(prob_Z)
    entropy_Q = compute_entropy(prob_Q)
    joint_entropy = compute_entropy(prob_matrix.flatten())
    mutual_information = entropy_Z + entropy_Q - joint_entropy
    return mutual_information, entropy_Z, entropy_Q


def compute_nmi(Z, Q):
    """
    Compute the Normalized Mutual Information (NMI) from ground truth clusters Z and predicted clusters Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    float: Normalized Mutual Information
    """
    contingency_matrix = create_contingency_matrix(Z, Q)
    mutual_information, entropy_Z, entropy_Q = compute_mutual_information(contingency_matrix)
    nmi = mutual_information / np.sqrt(entropy_Z * entropy_Q)
    return nmi


def compute_cosine_similarity(Z, Q):
    """
    Compute the Cosine Similarity from ground truth clusters Z and predicted clusters Q.
    Parameters:
    Z (list of sets): Ground truth clusters
    Q (list of sets): Predicted clusters
    Returns:
    float: Cosine Similarity
    """
    contingency_matrix = create_contingency_matrix(Z, Q)
    dot_product = np.sum(contingency_matrix ** 2)
    magnitude_Z = np.sqrt(np.sum(np.sum(contingency_matrix, axis=1) ** 2))
    magnitude_Q = np.sqrt(np.sum(np.sum(contingency_matrix, axis=0) ** 2))
    cosine_similarity = dot_product / (magnitude_Z * magnitude_Q)
    return cosine_similarity


# Example clusters Z and Q
Q = [{'o1', 'o3', 'o5', 'o9', 'o10', 'o8', 'o7'}, {'o4', 'o6'}, {'o2'}]
Z = [{'o1', 'o2', 'o3'}, {'o4', 'o5', 'o6', 'o7', 'o8'}, {'o9', 'o10'}]

# Calculate indices
rand_index = compute_rand_index(Z, Q)
jaccard_index = compute_jaccard_index(Z, Q)
smc = compute_smc(Z, Q)
nmi = compute_nmi(Z, Q)
cosine_similarity = compute_cosine_similarity(Z, Q)

print(f"Rand Index: {rand_index:.3f}")
print(f"Jaccard Index: {jaccard_index:.3f}")
print(f"Simple Matching Coefficient (SMC): {smc:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
print(f"Cosine Similarity: {cosine_similarity:.3f}")
```

---
## S23, Q8) Compare Rand and jaccard indexes #jaccard 
![[Pasted image 20240520203307.png]]

Just use script above but put Q clusters into 1 cluster.

---
## F23, Q17) Consider the observations and the pairwise distances in Table 6. At which height will groups/clusters containing o9 and o10 merge in a dendrogram when using the complete linkage function?

![[Pasted image 20240521005200.png]]

## Solution
Use this script to create the dendogram an check the height. 

```python
import numpy as np  # type: ignore  
import matplotlib.pyplot as plt  # type: ignore  
from scipy.cluster.hierarchy import dendrogram, linkage  # type: ignore  
from scipy.spatial.distance import squareform  # type: ignore  
  
correct_distance_matrix = np.array([  
    [0.0, 2.91, 0.63, 1.88, 1.02, 1.82, 1.92, 1.58, 1.08, 1.43],  
    [2.91, 0.0, 3.23, 3.9, 2.88, 3.27, 3.48, 4.02, 3.08, 3.47],  
    [0.63, 3.23, 0.0, 2.03, 1.06, 2.15, 2.11, 1.15, 1.09, 1.65],  
    [1.88, 3.9, 2.03, 0.0, 2.52, 1.04, 2.25, 2.42, 2.18, 2.17],  
    [1.02, 2.88, 1.06, 2.52, 0.0, 2.44, 2.38, 1.53, 1.71, 1.94],  
    [1.82, 3.27, 2.15, 1.04, 2.44, 0.0, 1.93, 2.72, 1.98, 1.8],  
    [1.92, 3.48, 2.11, 2.25, 2.38, 1.93, 0.0, 2.53, 2.09, 1.66],  
    [1.58, 4.02, 1.15, 2.42, 1.53, 2.72, 2.53, 0.0, 1.68, 2.06],  
    [1.08, 3.08, 1.09, 2.18, 1.71, 1.98, 2.09, 1.68, 0.0, 1.48],  
    [1.43, 3.47, 1.65, 2.17, 1.94, 1.8, 1.66, 2.06, 1.48, 0.0]  
])  
  
  
correct_distance_matrix = (correct_distance_matrix + correct_distance_matrix.T) / 2  
# Set diagonal to zero  
np.fill_diagonal(correct_distance_matrix, 0)  
  
# Convert the distance matrix to a condensed form  
condensed_matrix = squareform(correct_distance_matrix)  
  
# Perform hierarchical clustering using complete linkage  
  
# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' and 'average' for average in the second parameter  
linked = linkage(condensed_matrix, 'single')  
  
# Plot the dendrogram  
plt.figure(figsize=(10, 7))  
dendrogram(linked,  
           orientation='top',  
           labels=range(1, 11),  
           # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)  
           distance_sort='descending',  
           show_leaf_counts=True)  
plt.title('Dendrogram for Hierarchical Clustering')  
plt.xlabel('Index of Point')  
plt.ylabel('Distance')  
plt.show()
```

---

## F16, Q8) What is the above SMC between the true labeling of the observations into the three classes Kama, Rosa, and Canadian, and the clustering defined by thresholding Dendrogam 4 at the level of three clusters?

![[Pasted image 20240521150418.png]]

Script has been modified for other exam sets but  can be used for this question. 

```python
import numpy as np  # type: ignore  
  
# TODO: Assuming these cluster assignments from the dendrogram  
clusters = {  
    'cluster1': [7, 8, 9, 3],   
'cluster2': [1, 2],  
    'cluster3': [5, 4, 6]  
}  
  
clusters_s17 = {  
    'cluster1': [2, 3, 4, 6, 8],  
    'cluster2': [1, 5, 7]  
}  
  
# True class labels as per your description  
# TODO: look what class they are defined as:  
class_labels = {  
    1: 'K', 2: 'K', 3: 'K',    
4: 'R', 5: 'R', 6: 'R',    
7: 'C', 8: 'C', 9: 'C'  }  
  
class_labels_s17 = {  
    1: 'L', 2: 'L', 3: 'L',  
    4: 'L', 5: 'H', 6: 'H',  
    7: 'H', 8: 'L'  
}  
  
  
# Function to calculate f00 and f11  
def calculate_f00_f11(clusters, labels):  
    f00, f11 = 0, 0  
    all_observations = list(labels.keys())  
    K = len(all_observations) * (len(all_observations) - 1) // 2  
    for i in range(len(all_observations)):  
        for j in range(i + 1, len(all_observations)):  
            same_cluster = any(  
                all_observations[i] in clusters[cluster] and all_observations[j] in clusters[cluster] for cluster in  
                clusters)  
            same_class = labels[all_observations[i]] == labels[all_observations[j]]  
            if same_cluster and same_class:  
                f11 += 1  
            if not same_cluster and not same_class:  
                f00 += 1  
    return f00, f11, K  
  
  
# Calculate f00, f11, and K  
f00, f11, K = calculate_f00_f11(clusters_s17, class_labels_s17)  
  
# Calculate SMC  
SMC = (f00 + f11) / K  
  
# Calculate Jaccard coefficient  
jaccard = f11 / (K - f00)  
  
print(f"SMC: {SMC}")  
print(f"Jaccard coefficient: {jaccard}")  
print(f"Total pairs (K): {K}, f00: {f00}, f11: {f11}")
```

---

## S20, Q22)We assign the green observation to one of the three cluster by considering the proximity measure as computed based on Euclidean distance between the green point, and the points in the cluster. Which one of the following statements is correct?

![[Pasted image 20240521201646.png]]

The correct answer is A. Minimum linkage will result in the new observation assigned to red  crosses as the closest most observation is a red cross. Maximum linkage will assign the new observation to the blue plusses according to the furthest most observation of each cluster having a blue plus as closest to the new observation. Average linkage corresponds to considering the avarege distance and can be considered a center based approach. Here the center of the red crosses are furthest away and therefore this cannot be the cluster the new observation is assigned to. 