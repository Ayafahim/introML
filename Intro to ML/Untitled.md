

```python

# Q4  
  
# The provided V matrix as a numpy array  
V = np.array([  
    [0.49, -0.5,  0.08, -0.49,  0.52],  
    [0.58,  0.23, -0.01,  0.71,  0.33],  
    [0.56,  0.23,  0.43, -0.25, -0.62],  
    [0.31,  0.09, -0.9,  -0.19, -0.24],  
    [-0.06,  0.8,  0.03, -0.41,  0.43]  
])  
  
# We will go through the statements and check which one is true based on the signs of the weights  
  
# Statement A: Refers to PC5  
# Low value of Time of day (positive weight), low value of Broken Truck (positive weight),  
# high value of Accident victim (negative weight), high value of Immobilized bus (negative weight),  
# low value of Defects (positive weight) - This would typically have a negative projection on PC5.  
  
# Statement B: Refers to PC3  
# Low value of Accident victim (positive weight), high value of Immobilized bus (negative weight)  
# - This would typically have a negative projection on PC3.  
  
# Statement C: Refers to PC4  
# Low value of Time of day (negative weight), high value of Broken Truck (positive weight),  
# low value of Accident victim (negative weight), low value of Defects (negative weight)  
# - This would typically have a positive projection on PC4 because the high value of Broken Truck  
# with a positive weight would dominate.  
  
# Statement D: Refers to PC2  
# Low value of Time of day (negative weight), high value of Broken Truck (positive weight),  
# high value of Accident victim (positive weight), high value of Defects (positive weight)  
# - This would typically have a positive projection on PC2 because the positive weights dominate.  
  
# Now we'll check these statements against the weights in V to determine which statement is true.  
  
projection_A = -V[0,4] - V[1,4] + V[2,4] + V[3,4] - V[4,4] > 0  
projection_B = -V[2,2] + V[3,2] > 0  
projection_C = -V[0,3] + V[1,3] - V[2,3] - V[4,3] > 0  
projection_D = -V[0,1] + V[1,1] + V[2,1] + V[4,1] > 0  
  
var = projection_A, projection_B, projection_C, projection_D  
print(var)
```
