
## S19, Q18) ROC curve for table with binary classification and predicted class probability

![[Pasted image 20240516011012.png]]

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example data - replace these with your actual data from Table 5
# True binary labels (1 for positive class, 0 for negative class)
y_true = np.array([1, 1, 0, 1, 1, 1, 0])

# Predicted probabilities for the positive class
y_scores = np.array([0.14, 0.15, 0.27, 0.61, 0.71, 0.75, 0.81])

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure()
lw = 2  # Line width
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```


---

## F16, Q18)
![[Pasted image 20240521151827.png]]

![[Pasted image 20240521151831.png]]

```python
import numpy as np # type: ignore  
from sklearn.metrics import roc_curve, auc # type: ignore  
import matplotlib.pyplot as plt # type: ignore  
  
#TODO if table use this  
# Heights for each player based on your provided data  
table = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4])  
#TODO: Label is used if you have to fliter it in classes forexample red or black  
labels = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])  # Correct labels as you described  
  
#TODO use this if matrix, remember to read the task and see which one of the fields it is asking for  
#TODO if similar to exam16f Q18  
  
x1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0])  
y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])  
# Compute ROC curve and AUC  
  
  
# Data for gears and mpg classification  
#TODO: if table similar to exam17S Q11  
  
gears = np.array([3]*15 + [4]*12 + [5]*5)  
mpg = np.array([0]*13 + [1]*2 + [0]*2 + [1]*10 + [0]*2 + [1]*3)  
  
# # Simple numerical encoding for gears (although one-hot encoding might be more appropriate in other contexts)  
# numerical_gears = np.array([1 if gear == 4 else (2 if gear == 5 else 0) for gear in gears])  
  
  
# Extracted data from Figure 7 F18,Q13  
# True labels: 0 for copyist 1, 1 for copyist 2 and 3  
true_labels = np.array([0, 0, 1, 1, 0, 1, 0])  
# Predicted probabilities for the positive class (copyist 2 and 3)  
predicted_probs = np.array([0.1, 0.21, 0.4, 0.5, 0.55, 0.6, 0.61])  
  
  
  
#TODO: Update roc_curve based on the task  
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)  
roc_auc = auc(fpr, tpr)  
  
# Plotting the ROC Curve  
plt.figure(figsize=(8, 6))  
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver Operating Characteristic for Player Height vs FG%')  
plt.legend(loc="lower right")  
plt.grid(True)  
plt.show()
```

---
## S23,  Q21)



The problem involves identifying which ROC curve corresponds to the given true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) as a function of the threshold $\theta$.

#### Given Information:

- **TP, TN, FP, and FN Curves**: These are plotted as functions of the threshold $ \theta $.
- **ROC Curves**: Four candidate ROC curves are provided.

### Approach to Solution:

1. **Identify Total Positives and Negatives**:
   - The leftmost value of the TP curve gives the total number of positive-class observations $P$.
   - The rightmost value of the TN curve gives the total number of negative-class observations $N$.

2. **Choose a Point on the ROC Curve**:
   - Select a threshold $\theta$ (commonly $\theta = 0.5$ is used).
   - Determine the number of FP and TP at this threshold.

3. **Calculate False Positive Rate (FPR) and True Positive Rate (TPR)**:
   - $\text{FPR} = \frac{\text{FP}}{N}$
   - $\text{TPR} = \frac{\text{TP}}{P}$

4. **Match with Candidate ROC Curves**:
   - Identify the ROC curve that includes the calculated point $(\text{FPR}, \text{TPR})$.

### Detailed Solution Steps:

1. **From the Solution Document**:

   - **Total Positives (P)**: From the TP curve (left-most value), $P = 146$.
   - **Total Negatives (N)**: From the TN curve (right-most value), $N = 136$.

2. **Select $\theta = 0.5$**:

   - At $\theta = 0.5$, the values are given as:
     - $text{FP} = 42$
     - $\text{TP} = 123$

3. **Calculate FPR and TPR**:

   - **FPR**: 
     $$
     \text{FPR} = \frac{\text{FP}}{N} = \frac{42}{136} \approx 0.31
     $$

   - **TPR**: 
     $$
     \text{TPR} = \frac{\text{TP}}{P} = \frac{123}{146} \approx 0.84
     $$

4. **Matching with ROC Curves**:

   - From the solution, the point $(0.31, 0.84)$ rules out all options except ROC Curve 4.
### Why?

- **ROC Curve Definition**: The ROC curve plots the TPR versus FPR at various threshold settings. A specific point on this curve represents the performance of the classifier at a particular threshold.
- **Point (0.31,0.84)**: This point should appear on the correct ROC curve corresponding to the threshold 𝜃=0.5.

### Conclusion:

The correct ROC curve corresponding to the given TP, TN, FP, and FN functions is:

**D. ROC Curve 4**

```python
import numpy as np

# Given data
P = 146  # Total positive observations
N = 136  # Total negative observations

# Values at theta = 0.5
TP = 123
FP = 42

# Calculate FPR and TPR
FPR = FP / N
TPR = TP / P

print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"True Positive Rate (TPR): {TPR:.2f}")
```

