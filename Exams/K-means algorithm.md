
## S19, Q12) What will the location of the cluster centers be after the k-means algorithm has converged? #convergance
![[Pasted image 20240515225447.png]]
```python
import numpy as np

# Data and initial centroids
x = np.array([1.0, 1.2, 1.8, 2.3, 2.6, 3.4, 4.0, 4.1, 4.2, 4.6])
initial_centroids = np.array([1.8, 3.3, 3.6])
K = 3  # Number of clusters

def k_means(x, centroids, K):
    converged = False
    while not converged:
        # Assign points to the nearest centroid
        distances = np.abs(x[:, np.newaxis] - centroids[np.newaxis, :])
        closest_centroid = np.argmin(distances, axis=1)
        
        # Recalculate centroids
        new_centroids = np.array([x[closest_centroid == k].mean() for k in range(K)])
        
        # Check convergence
        if np.allclose(new_centroids, centroids):
            converged = True
        centroids = new_centroids
    
    return centroids

final_centroids = k_means(x, initial_centroids, K)
print("Final centroids:", final_centroids)
```

---

## S23, Q9) What will be the total cost (sum of squared distances), E, after the first iteration of K-means?
![[Pasted image 20240520203528.png]]

## Script

```python
import numpy as np  
  
# TODO Data points  
data = np.array([2, 5, 8,12, 13 ])  
  
# TODO Initial centroids based on the first three observations  
centroids = np.array([4, 10])  
  
# Function to assign points to the nearest centroid  
def assign_points_to_centroids(data, centroids):  
    clusters = {}  
    for point in data:  
        # Calculate distance from the point to each centroid  
        distances = np.abs(point - centroids)  
        # Find the nearest centroid (index of the minimum distance)  
        nearest = np.argmin(distances)  
        if nearest not in clusters:  
            clusters[nearest] = []  
        clusters[nearest].append(point)  
    return clusters  
  
# Function to recalculate centroids  
def recalculate_centroids(clusters):  
    new_centroids = np.array([np.mean(clusters[key]) for key in sorted(clusters.keys())])  
    return new_centroids  
  
# Function to calculate the sum of squared distances  
def calculate_sum_of_squared_distances(clusters, centroids):  
    sum_of_squared_distances = 0  
    for key in clusters:  
        centroid = centroids[key]  
        for point in clusters[key]:  
            sum_of_squared_distances += (point - centroid) ** 2  
    return sum_of_squared_distances  
  
# Perform the k-means clustering algorithm  
previous_centroids = None  
iteration = 0  
while not np.array_equal(centroids, previous_centroids):  
    print(f"Iteration {iteration}, Centroids: {centroids}")  
    clusters = assign_points_to_centroids(data, centroids)  
    E = calculate_sum_of_squared_distances(clusters, centroids)  
    print(f"Sum of squared distances (E): {E}")  
    previous_centroids = centroids  
    centroids = recalculate_centroids(clusters)  
    iteration += 1  
  
# Final clusters  
final_clusters = {key: clusters[key] for key in sorted(clusters.keys())}  
print("Final Clusters:", final_clusters)
```
