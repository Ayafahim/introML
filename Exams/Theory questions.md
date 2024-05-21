
## F23, Q4) #PCA 

![[Pasted image 20240520212754.png]]

---
## F23, Q5)
![[Pasted image 20240520213428.png]]

---
## F23,Q10) Which one of the following statements regarding artificial neural networks is correct? #ANN 

![[Pasted image 20240518003857.png]]
---
## F23, Q14) #AdaBoost 

![[Pasted image 20240520220421.png]]

---
## F23, Q25) #Apriori #associationMining

![[Pasted image 20240521015004.png]]
![[Pasted image 20240521015057.png]]

---

![[Pasted image 20240521015126.png]]

---
## F16, 26) For which of the following purposes is cross-validation the least well suited? #crossvalidation
![[Pasted image 20240521153638.png]]

Cross-validation can trivially be used to quantify the number of hidden units in ANN and nearest neighbors in KNN classification by evaluating the performance predicting the output in these supervised learning problems. As we have seen in the course crossvalidation can also be used to quantify the width of the kernel density estimator. Cross-validation is used to quantify models generalization through the use of the test sets and not to minimize the training error.

---
## F16, 27) Which of the following statements regarding ensemble methods is correct?

![[Pasted image 20240521153808.png]]

---

## F17, Q17) #crossvalidation 

![[Pasted image 20240521155842.png]]

---
## F17, Q26)

![[Pasted image 20240521162147.png]]


Multinomial regression is a generalization of two class logistic regression to handle multiple classes. Decision trees do not return probabilities of being in each class but hard assigns observations to the classes based on majority voting in each terminal leaf. K-means, Gaussian Mixture Models and Artificial Neural Networks (ANN) are indeed all prone to local minima and it is therefore advised to use multiple restarts selecting the initialization with best soution. Accuracy is not a good performance measure when facing severe class-imbalance issues as we may trivially obtain a very high accuracy simply by classifying by chance. The AUC of the receiver operator characteristic would here be more appropriate as it is not influenced by the relative sizes of the two classes.

---
## S20, Q26) #crossvalidation 

![[Pasted image 20240521203102.png]]

---
## S22, Q11)
Question 11 Explanation
The ROC (Receiver Operating Characteristic) curve is a plot of the True Positive Rate (TPR) against the False Positive Rate (FPR) for different threshold values. The AUC is a single scalar value that summarizes the performance of the classifier; it represents the area under the ROC curve.

Key concepts:

![[Pasted image 20240521223053.png]]


Analyzing the Statements
We need to evaluate which of the statements about the ROC curve and AUC is true based on the provided information.

Statement A: "The ROC curve shows how an increase in the accuracy results in a (monotonic) increase in true positive rate."

This statement is incorrect. The ROC curve does not directly show accuracy. It plots TPR against FPR for various thresholds.
Statement B: "A classifier having a ROC curve that passes through (0.0, 1.0) can achieve an FPR and FNR of zero on the particular dataset."

This statement is incorrect. If the ROC curve passes through (0.0, 1.0), it means there is a threshold for which the classifier achieves 100% TPR and 0% FPR, but it does not guarantee zero FNR (False Negative Rate).
Statement C: "A classifier having a ROC curve with an AUC of 0.5 implies that the FPR= 0.5 and the FNR= 0.5."

This statement is incorrect. An AUC of 0.5 implies that the classifier performs no better than random guessing, but it does not mean that FPR and FNR are both 0.5.
Statement D: "A classifier with an AUC of 0.5 implies that the ROC curve passes through (0.0, 0.0), (0.5, 0.5), and (1.0, 1.0)."

This statement is correct. An AUC of 0.5 implies that the classifier performs like random guessing, and the ROC curve would be a diagonal line from (0.0, 0.0) to (1.0, 1.0), passing through (0.5, 0.5).
Based on the analysis, the correct answer is: 𝐷


Summary
An ROC curve with an AUC of 0.5 implies a diagonal line from (0.0, 0.0) to (1.0, 1.0), indicating that the classifier's performance is equivalent to random guessing. This is the essence of Statement D.