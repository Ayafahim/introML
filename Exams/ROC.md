
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
