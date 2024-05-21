
## F16, Q6) What is the accuracy of the classifier?

![[Pasted image 20240521145910.png]]

![[Pasted image 20240521145933.png]]


```python
import numpy as np # type: ignore  

  
def compute_metrics(confusion_matrix):  
    # Calculate the number of correct predictions (sum of diagonal elements)  
    correct_predictions = np.trace(confusion_matrix)  
  
    # Calculate the total number of predictions (sum of all elements)  
    total_predictions = np.sum(confusion_matrix)  
  
    # Calculate accuracy  
    accuracy = correct_predictions / total_predictions  
  
    # Initialize arrays to store precision and recall for each class  
    precision = np.zeros(confusion_matrix.shape[0])  
    recall = np.zeros(confusion_matrix.shape[0])  
  
    # Calculate precision and recall for each class  
    for i in range(confusion_matrix.shape[0]):  
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) if np.sum(confusion_matrix[:, i]) != 0 else 0  
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) if np.sum(confusion_matrix[i, :]) != 0 else 0  
  
    return accuracy, precision, recall  
  
# Example of a 3x3 confusion matrix  
confusion_matrix_3x3 = np.array([  
    [31, 1, 3],  # Predictions for Class 1  
    [5, 30, 0],  # Predictions for Class 2  
    [6, 0, 29]   # Predictions for Class 3  
])  
  
confusion_matrix_2x2 = np.array([  
    [36, 15],  # Predictions for Class 1  
    [10, 39]  # Predictions for Class 2  
])  
  
# Calculate metrics  
accuracy, precision, recall = compute_metrics(confusion_matrix_2x2)  
  
# Print the results  
print(f"The accuracy of the classifier is: {accuracy:.4f}")  
print("Precision for each class:", precision)  
print("Recall for each class:", recall)
```

---
## S22, Q16)

![[Pasted image 20240521232051.png]]

To solve Question 16, we need to analyze the confusion matrices provided for the logistic regression (LR) classifier and the decision tree (DT) classifier on the Fish Toxicity dataset. The goal is to determine which classifier is preferable based on certain performance criteria.

### Given Confusion Matrices

#### Decision Tree (DT)
```
Predicted
       low    high
Actual
low    371     93
high    97    347
```

#### Logistic Regression (LR)
```
Predicted
       low    high
Actual
low    359    141
high    39    369
```

### Analysis

#### Accuracy
- **DT Accuracy**: $\frac{371 + 347}{908} = \frac{718}{908} \approx 0.791$
- **LR Accuracy**: $\frac{359 + 369}{908} = \frac{728}{908} \approx 0.802$

#### False Positive Rate (FPR)
- **DT FPR**: $\frac{93}{464} \approx 0.200$
- **LR FPR**: $\frac{141}{464} \approx 0.304$

#### False Negative Rate (FNR)
- **DT FNR**: $\frac{97}{444} \approx 0.218$
- **LR FNR**: $\frac{39}{444} \approx 0.088$

### Objective

The objective is to avoid wrongly classifying a chemical as having a low NLLC50 value (i.e., high toxicity) when it is actually high. This means we want to minimize the False Negative Rate (FNR).

### Evaluating Options

1. **Option A**: The LR classifier is the preferred option because it has a slightly better accuracy compared to the DT classifier, and the LR classifier’s FPR is lower than the DT classifier’s FPR.
   - This statement is incorrect because the LR classifier has a higher FPR compared to the DT classifier.

2. **Option B**: The LR classifier is the preferred option because it only has a slightly worse accuracy compared to the DT classifier, and the LR classifier’s FNR is lower than the DT classifier’s FNR.
   - This statement is incorrect because the LR classifier has a slightly better accuracy, not worse.

3. **Option C**: The DT classifier is the preferred option because it has a slightly better accuracy compared to the LR classifier, and the DT classifier’s FNR is lower than the LR classifier’s FNR.
   - This statement is incorrect because the DT classifier has a slightly worse accuracy and a higher FNR compared to the LR classifier.

4. **Option D**: The DT classifier is the preferred option because it only has a slightly worse accuracy compared to the LR classifier, and the DT classifier’s FPR is lower than the LR classifier’s FPR.
   - This statement is correct because the DT classifier has a slightly worse accuracy but a significantly lower FPR compared to the LR classifier, which is crucial for the given objective.

Thus, the correct answer is:

$$ \boxed{\text{D}} $$

The DT classifier is the preferred option because it only has a slightly worse accuracy compared to the LR classifier, and the DT classifier’s FPR is lower than the LR classifier’s FPR.