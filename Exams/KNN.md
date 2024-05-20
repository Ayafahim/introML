
## S23, Q5) What is the density for observation o1 for K = 3 nearest neighbors?

![[Pasted image 20240520194024.png]]

### Solution 

Just find the 3 nearest neighbours and put them into formula or script below.

![[Pasted image 20240520194312.png]]


### Script either with whole table or just the values:

```python
import numpy as np  # type: ignore  
  
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
  
  
# Function to calculate the density based on the provided formula  
def calculate_density(distances, K):  
    return 1 / np.mean(np.sort(distances)[1:K + 1])  # Exclude the zero distance (self)  
  
  
# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)  
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python  
density_o4 = calculate_density(correct_distance_matrix[3], K=2)  
density_o6 = calculate_density(correct_distance_matrix[5], K=2)  
density_o1 = calculate_density(correct_distance_matrix[0], K=2)  
  
print("Densities of o2, o5 and o10:")  
print(density_o4, density_o6, density_o1)  
  
ard_o2_solution = density_o4 / ((density_o6 + density_o1) / 2)  
  
print("ARD: " + str(ard_o2_solution))  
  
  
# To make the script more readable, let's define a function that creates the proximity matrix and prints it in a readable format.  
  
def create_and_print_proximity_matrix(distance_matrix):  
    # Invert the distances to create proximity values, avoid division by zero  
    proximity_matrix = 1 / np.where(distance_matrix > 0, distance_matrix, np.inf)  
    # Set the diagonal to zeros  
    np.fill_diagonal(proximity_matrix, 0)  
  
    # Convert the proximity matrix to a string with rounded values for better readability  
    proximity_matrix_str = np.array2string(proximity_matrix, precision=3)  
  
    print("Proximity Matrix:\n" + proximity_matrix_str)  
  
  
# Call the function with the corrected distance matrix  
print(create_and_print_proximity_matrix(correct_distance_matrix))
```


### Without the table

```python
import numpy as np  # type: ignore  
  
  
def calculate_density(nearest_distances, K):  
    """    
    Calculate the density for a given observation based on its K-nearest neighbors.    
    Parameters:    - nearest_distances: A list of distances from the observation to its K-nearest neighbors.    - K: The number of nearest neighbors to consider.    
Returns:    - The density value for the observation.    """  # Ensure the list is sorted and take the K smallest distances  
    if len(nearest_distances) > K:  
        nearest_distances = sorted(nearest_distances)[:K]  
    return 1 / np.mean(nearest_distances)  
  
  
def calculate_ard(nearest_distances_o, nearest_distances_neighbors, K):  
    """  
    Calculate the average relative density (a.r.d.) for a given observation.    Parameters:    - nearest_distances_o: A list of distances from the observation to its K-nearest neighbors.    - nearest_distances_neighbors: A list of lists, each containing distances from each neighbor to its K-nearest neighbors.    - K: The number of nearest neighbors to consider.    Returns:    - The average relative density (a.r.d.) for the observation.    """    density_o = calculate_density(nearest_distances_o, K)  
    print(f"Density for class: {density_o:.4f}")  
    densities_neighbors = [calculate_density(distances, K) for distances in nearest_distances_neighbors]  
    return density_o / np.mean(densities_neighbors)  
  
  
# Example usage  
  
# Remember whatever K is, is the amount of neighbours to find. fo remember to add them.  
K = 2  
nearest_distances_o1 = [1.3, 2.4, 2.7]  # Distances to the two nearest neighbors o2 and o5  
nearest_distances_o2 = [1.3, 2.2, 2.3]  # Distances from o6 to its two nearest neighbors  
nearest_distances_o6 = [0.2608, 1.5155]  # Distances from o4 to its two nearest neighbors  
nearest_distances_o8 = [1.8870, 1.8926]  # Distances from o4 to its two nearest neighbors  
nearest_distances_o4 = [1.4852, 1.8926]  # Distances from o4 to its two nearest neighbors  
  
# Calculate a.r.d. for o9  
ard_o7 = calculate_ard(nearest_distances_o8, [nearest_distances_o6,nearest_distances_o4], K)  
  
# Print the result  
print(f"Average Relative Density (a.r.d.) of o7: {ard_o7:.4f}")
```


---

## F23, Q15) redict the label of the test point $o_test = [0   0]^T$

![[Pasted image 20240521003452.png]]

### Breakdown:
To solve this problem, we need to understand how each distance metric works and then apply the KNN algorithm with $K = 3$ to predict the label for the test point $[0, 0]^T$. Let's break down each step:

1. **Understanding Distance Metrics:**
   - $d_2$ (Euclidean Distance): This is the standard distance calculated as:
     $$
     d_2(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
     $$
   - $d_1$ (Manhattan Distance): This is the sum of absolute differences:
     $$
     d_1(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^n |x_i - y_i|
     $$
   - $d_\infty$ (Chebyshev Distance): This is the maximum absolute difference:
     $$
     d_\infty(\mathbf{x}, \mathbf{y}) = \max_i |x_i - y_i|
     $$

2. **Synthetic Dataset:**
   - The dataset includes several points with coordinates and their corresponding classes. The exact coordinates and classes from Figure 5 need to be known to proceed with calculations. For the sake of this explanation.
   -
1. **Calculate Distances:**
   - Calculate the distances from the test point $[0, 0]$ to each of the points in the dataset using $d_2$, $d_1$, and $d_\infty$.

2. **Apply KNN with $K = 3$:**
   - Identify the 3 nearest neighbors based on each distance metric.
   - Determine the class of the test point by majority voting among these 3 neighbors.

### Detailed Calculations:

#### Euclidean Distance ($d_2$):
$$ d_2(o_{\text{test}}, o_1) = \sqrt{(-0.4 - 0)^2 + (-0.8 - 0)^2} = \sqrt{0.16 + 0.64} = \sqrt{0.8} \approx 0.894 $$
$$ d_2(o_{\text{test}}, o_2) = \sqrt{(-0.9 - 0)^2 + (0.3 - 0)^2} = \sqrt{0.81 + 0.09} = \sqrt{0.9} \approx 0.949 $$
$$ d_2(o_{\text{test}}, o_3) = \sqrt{(0 - 0)^2 + (0.9 - 0)^2} = \sqrt{0.81} \approx 0.9 $$
$$ d_2(o_{\text{test}}, o_4) = \sqrt{(1 - 0)^2 + (-0.1 - 0)^2} = \sqrt{1 + 0.01} = \sqrt{1.01} \approx 1.005 $$
$$ d_2(o_{\text{test}}, o_5) = \sqrt{(0.8 - 0)^2 + (-0.7 - 0)^2} = \sqrt{0.64 + 0.49} = \sqrt{1.13} \approx 1.063 $$
$$ d_2(o_{\text{test}}, o_6) = \sqrt{(0.1 - 0)^2 + (0.8 - 0)^2} = \sqrt{0.01 + 0.64} = \sqrt{0.65} \approx 0.806 $$

The three nearest neighbors by $d_2$ are $o_6$, $o_1$, and $o_3$.

#### Manhattan Distance ($d_1$):
$$ d_1(o_{\text{test}}, o_1) = | -0.4 - 0 | + | -0.8 - 0 | = 0.4 + 0.8 = 1.2 $$
$$ d_1(o_{\text{test}}, o_2) = | -0.9 - 0 | + | 0.3 - 0 | = 0.9 + 0.3 = 1.2 $$
$$ d_1(o_{\text{test}}, o_3) = | 0 - 0 | + | 0.9 - 0 | = 0.9 $$
$$ d_1(o_{\text{test}}, o_4) = | 1 - 0 | + | -0.1 - 0 | = 1 + 0.1 = 1.1 $$
$$ d_1(o_{\text{test}}, o_5) = | 0.8 - 0 | + | -0.7 - 0 | = 0.8 + 0.7 = 1.5 $$
$$ d_1(o_{\text{test}}, o_6) = | 0.1 - 0 | + | 0.8 - 0 | = 0.1 + 0.8 = 0.9 $$

The three nearest neighbors by $d_1$ are $o_3$, $o_6$, and $o_4$.

#### Chebyshev Distance ($d_\infty$):
$$ d_\infty(o_{\text{test}}, o_1) = \max(| -0.4 - 0 |, | -0.8 - 0 |) = \max(0.4, 0.8) = 0.8 $$
$$ d_\infty(o_{\text{test}}, o_2) = \max(| -0.9 - 0 |, | 0.3 - 0 |) = \max(0.9, 0.3) = 0.9 $$
$$ d_\infty(o_{\text{test}}, o_3) = \max(| 0 - 0 |, | 0.9 - 0 |) = 0.9 $$
$$ d_\infty(o_{\text{test}}, o_4) = \max(| 1 - 0 |, | -0.1 - 0 |) = \max(1, 0.1) = 1 $$
$$ d_\infty(o_{\text{test}}, o_5) = \max(| 0.8 - 0 |, | -0.7 - 0 |) = \max(0.8, 0.7) = 0.8 $$
$$ d_\infty(o_{\text{test}}, o_6) = \max(| 0.1 - 0 |, | 0.8 - 0 |) = \max(0.1, 0.8) = 0.8 $$

The three nearest neighbors by $d_\infty$ are $o_1$, $o_5$, and $o_6$.

### Classification by KNN:
- **Euclidean Distance ($d_2$)**: Nearest neighbors are $o_6$ (C2), $o_1$ (C1), and $o_3$ (C1). Majority class is C1.
- **Manhattan Distance ($d_1$)**: Nearest neighbors are $o_3$ (C1), $o_6$ (C2), and $o_4$ (C2). Majority class is C2.
- **Chebyshev Distance ($d_\infty$)**: Nearest neighbors are $o_1$ (C1), $o_5$ (C2), and $o_6$ (C2). Majority class is C2.

### Conclusion:
From the analysis, we can see:
- $d_2$ predicts C1.
- $d_1$ predicts C2.
- $d_\infty$ predicts C2.

Therefore, the correct answer is:
C. $d_1$ and $d_\infty$ predict the same label.

### Script

```python
import numpy as np
from collections import Counter

# Define the distance functions
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def chebyshev_distance(point1, point2):
    return np.max(np.abs(point1 - point2))

# Function to calculate distances
def calculate_distances(test_point, dataset, distance_func):
    distances = []
    for point in dataset:
        distances.append(distance_func(test_point, np.array(point[:2])))
    return distances

# KNN classifier function
def knn_classifier(test_point, dataset, k, distance_func):
    distances = calculate_distances(test_point, dataset, distance_func)
    sorted_indices = np.argsort(distances)
    nearest_neighbors = [dataset[i] for i in sorted_indices[:k]]
    classes = [neighbor[2] for neighbor in nearest_neighbors]
    majority_class = Counter(classes).most_common(1)[0][0]
    return majority_class

# Define the dataset
# Format: [x1, x2, class]
dataset = [
    [-0.4, -0.8, 'C1'],
    [-0.9, 0.3, 'C1'],
    [0, 0.9, 'C1'],
    [1, -0.1, 'C2'],
    [0.8, -0.7, 'C2'],
    [0.1, 0.8, 'C2']
]

# Define the test point and value of k
test_point = np.array([0, 0])
k = 3

# Perform KNN classification for different distance metrics
euclidean_class = knn_classifier(test_point, dataset, k, euclidean_distance)
manhattan_class = knn_classifier(test_point, dataset, k, manhattan_distance)
chebyshev_class = knn_classifier(test_point, dataset, k, chebyshev_distance)

# Print the results
print(f"Classification using Euclidean distance (d2): {euclidean_class}")
print(f"Classification using Manhattan distance (d1): {manhattan_class}")
print(f"Classification using Chebyshev distance (d∞): {chebyshev_class}")

# Easily modifiable sections
# To change the dataset, modify the 'dataset' variable.
# To change the test point, modify the 'test_point' variable.
# To change the value of k, modify the 'k' variable.
# The distance functions and KNN classifier function can also be customized as needed.
```
