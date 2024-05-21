

## S23,  Q15) What is the correct rule assignment to the nodes in the decision tree?

![[Pasted image 20240521025217.png]]

![[Pasted image 20240521025237.png]]


## Solution

Use script below to plot the rules. Here are options A, B & D plotted, you can see that the areas get darker, and those are the one to mach to the plot given in figure 6, therefore we can see it is plot 1 which is option which is actually option B (i know confusing, but i added them in a stupid order when testing. You can decide your own order.) 

![[Pasted image 20240521025142.png]]

## Script

NOTE: if the threshold for some reason changes to > instead of < or == or something else just change that in the  `plot_decision_region`function.
Also, chebyshev = infinity :)

```python
import numpy as np  
import matplotlib.pyplot as plt  
  
def calculate_distance(X, Y, point, distance_type):  
    """  
    Calculate the distance based on the specified distance type.  
    Parameters:    X, Y: Meshgrid arrays.    point: Tuple (x, y) representing the point to calculate the distance from.    distance_type (str): Type of distance metric ('euclidean', 'manhattan', 'chebyshev', or 'abs').  
    Returns:    Distance array.    """    if distance_type == 'euclidean':  
        return np.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2)  
    elif distance_type == 'manhattan' or distance_type == 'abs':  
        return np.abs(X - point[0]) + np.abs(Y - point[1])  
    elif distance_type == 'chebyshev':  
        return np.maximum(np.abs(X - point[0]), np.abs(Y - point[1]))  
    else:  
        raise ValueError("Unsupported distance type. Choose 'euclidean', 'manhattan', 'chebyshev', or 'abs'.")  
  
def plot_decision_region(ax, distance_types, points, thresholds, grid_size=400):  
    """  
    Plots decision regions based on specified distance metrics, points, and thresholds on a given axis.  
    Parameters:    ax: Matplotlib axis to plot on.    distance_types (tuple): Distance types for each region (A, B, C, D).    points (tuple): Points for each region (A, B, C, D).    thresholds (tuple): Thresholds for each region (A, B, C, D).    grid_size (int): Number of points in the grid.    """    # Create the grid  
    x = np.linspace(0, 6, grid_size)  
    y = np.linspace(0, 6, grid_size)  
    X, Y = np.meshgrid(x, y)  
  
    # Calculate the distance for each region  
    A = calculate_distance(X, Y, points[0], distance_types[0]) < thresholds[0]  
    B = calculate_distance(X, Y, points[1], distance_types[1]) < thresholds[1]  
    C = calculate_distance(X, Y, points[2], distance_types[2]) < thresholds[2]  
    D = calculate_distance(X, Y, points[3], distance_types[3]) < thresholds[3]  
  
    # Plotting  
    ax.imshow(A, extent=(0, 6, 0, 6), origin='lower', cmap='Reds', alpha=0.3)  
    ax.imshow(B, extent=(0, 6, 0, 6), origin='lower', cmap='Greens', alpha=0.3)  
    ax.imshow(C, extent=(0, 6, 0, 6), origin='lower', cmap='Blues', alpha=0.3)  
    ax.imshow(D, extent=(0, 6, 0, 6), origin='lower', cmap='Purples', alpha=0.3)  
  
    ax.set_xlabel('x1')  
    ax.set_ylabel('x2')  
    ax.grid(True)  
  
def plot_multiple_decision_regions(list_of_sets, grid_size=400):  
    """  
    Plots multiple sets of decision regions side by side for comparison.  
    Parameters:    list_of_sets (list): List of tuples where each tuple contains distance_types, points, and thresholds.    grid_size (int): Number of points in the grid.    """    num_plots = len(list_of_sets)  
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))  
  
    for idx, (distance_types, points, thresholds) in enumerate(list_of_sets):  
        plot_decision_region(axs[idx], distance_types, points, thresholds, grid_size)  
        axs[idx].set_title(f"Plot {idx+1}")  
  
    plt.tight_layout()  
    plt.show()  
  
# Define your sets of rules  
set_1 = (  
    ('euclidean', 'manhattan', 'manhattan', 'euclidean'),  
    ((6, 2), (2, 6), (2, 4), (2, 4)),  
    (3, 3, 3, 2)  
)  
  
set_2 = (  
    ('manhattan', 'manhattan', 'euclidean', 'euclidean'),  
    ((2, 4), (2, 6), (6, 2), (2, 4)),  
    (3, 3, 3, 2)  
)  
set_4 = (('euclidean', 'manhattan', 'euclidean', 'manhattan'),  
         ((2, 4), (2, 6), (6, 2), (2, 4)),  
         (2,3,3,3))  
  
# Plot the sets side by side  
plot_multiple_decision_regions([set_1, set_2, set_4])  
  
# To add more sets, simply include them in the list passed to plot_multiple_decision_regions  
# For example:  
# set_3 = (distance_types_3, points_3, thresholds_3)  
# set_4 = (distance_types_4, points_4, thresholds_4)  
# plot_multiple_decision_regions([set_1, set_2, set_3, set_4])
```


## classify the point, pick any random point and classify it

example for S19, Q21

![[Pasted image 20240521143859.png]]

![[Pasted image 20240521143904.png]]

```python
import numpy as np  
  
# Example classification tree based on the provided image and rules  
# inf: np.maximum(np.abs(X - point[0]), np.abs(Y - point[1]))  
# 1-norm: np.abs(X - point[0]) + np.abs(Y - point[1])  
# 2-norm: np.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2)  
  
classification_tree = {  
    'A': {  
        'condition': lambda x: np.abs(x[0] - 2) + np.abs(x[1] - 4) < 2,  # L1 norm for node A  
        'false': {  
            'B': {  
                'condition': lambda x: np.linalg.norm([x[0] - 6, x[1]], ord=2) < 3,  # L2 norm for node B  
                'true': 'Class 3',  
                'false': 'Class 2'  
            }  
        },  
        'true': {  
            'C': {  
                'condition': lambda x: np.linalg.norm([x[0] - 4, x[1] - 2], ord=2) < 2,  # L2 norm for node C  
                'true': 'Class 2',  
                'false': 'Class 1'  
            }  
        }  
    }  
}  
  
def classify(tree, point):  
    """  
    Traverse the classification tree to classify the given point.    """    if isinstance(tree, dict):  
        node = list(tree.keys())[0]  
        condition = tree[node]['condition']  
        if condition(point):  
            return classify(tree[node]['true'], point)  
        else:  
            return classify(tree[node]['false'], point)  
    else:  
        return tree  
  
# Example usage of the classification function  
example_point = [2.5, 0.5]  
classified_class = classify(classification_tree, example_point)  
print(f"Classified class for point {example_point}: {classified_class}")
```