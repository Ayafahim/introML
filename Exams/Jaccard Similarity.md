
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

