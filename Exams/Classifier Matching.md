
## F23, Q20) Match the model to the classifier

![[Pasted image 20240521005851.png]]


For these plots ANN usually has a hump, its mostly smooth but can be complex. An artificial neural network with few hidden units can have some non-linearity, but otherwise have boundaries of limited complexity and consisting of relatively simple shapes Classifier 3 is ANN. 

CT splits the feature space into rectangles or hyperrectangles. Boundaries are aligned with the feature axes. Look for straight lines that are perpendicular to the feature axes. The decision regions will look like rectangles or boxes. A decision tree has axis aligned splits, therefore  
the boundaries must be vertical or horizontal. Classifier 2 is CT.

Logistic Regression has a clear distinction between two classes, typically indicated by different colors or shades. The decision boundary is a straight line (or hyperplane in higher dimensions). A single linear line separating two regions. So classifier 1. 

For KNN the boundaries can be quite jagged, reflecting the local structure of the data. Regions around each data point that show the influence of the nearest neighbors. So classifier 4.

---
## F23, Q23) Consider a two-dimensional data set comprised of N = 9 observations shown in Figure 14.  The dataset consists of three classes indicated by the blue squares (class 1), red triangles (class 2) and yellow circles (class 3). In the figure, the decision boundaries for four K-nearest neighbor classifiers (KNN) are shown. Which one of the plots correspond to the K = 3 nearest-neighbour classifier assuming ties are broken by assigning to the nearest neighbour’s class? #KNN 

![[Pasted image 20240521013248.png]]


## Solution

A point to the left of the figure must clearly be assigned to class 2 because of the two triangles. A point furthest to the right will have three different classes in its K = 3 nearest neighborhood and the tie is broken by assigning it to class 2 as well. The rules out all options except A and B. These differ only in the assignment of a point near the left-most blue square, but a point here will have two yellow circles inits K = 3 neighborhood and therefore belong to class 3. This rules out all options except D.

Remember if there is a tie it is assigned to the closest one.

--- 
## F16, Q5)

![[Pasted image 20240521145617.png]]

---
# F17, Q15)
![[Pasted image 20240521155743.png]]