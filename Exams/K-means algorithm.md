

## S19, Q12) What will the location of the cluster centers be after the k-means algorithm has converged?
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

