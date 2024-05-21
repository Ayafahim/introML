

## S19, Q19) Consider again the travel review dataset in Table 1. We would like to predict a resort’s rating using a linear regression, and since we would like the model to be as interpretable as possible we will use variable selection to obtain a parsimonious model. We limit ourselves to the five features x1, x6, x7, x8, x9 and in Table 6 we have pre-computed the estimated training and test error for different variable combinations of the dataset. Which of the following statements is correct?

![[Pasted image 20240521132749.png]]


### Solution

> [!NOTE]
> Remember picking the 'the best feature' means picking the lowest value



The correct answer is **B**. To solve this problem, it suffices to show which variables will be selected by forward/backward selection. First note that in variable selection, we only need concern ourselves with the *test error*, as the training error should as a rule trivially drop when more variables are introduced and is furthermore not what we ultimately care about.

### Forward selection: 
The method is initialized with the set ${}$ having an error of 5.528.

#### Step $i = 1$
The available variable sets to choose between is obtained by taking the current variable set ${}$ and adding each of the left-out variables thereby resulting in the sets ${x_1}$, ${x_6}$, ${x_7}$, ${x_8}$, ${x_9}$. Since the lowest error of the available sets is 4.57, which is lower than 5.528, we update the current selected variables to ${x_6}$.

#### Step $i = 2$
The available variable sets to choose between is obtained by taking the current variable set ${x_6}$ and adding each of the left-out variables thereby resulting in the sets ${x_1, x_6}$, ${x_1, x_7}$, ${x_6, x_7}$, ${x_1, x_8}$, ${x_6, x_8}$, ${x_7, x_8}$, ${x_1, x_9}$, ${x_6, x_9}$, ${x_7, x_9}$, ${x_8, x_9}$. Since the lowest error of the available sets is 4.213, which is lower than 4.57, we update the current selected variables to ${x_1, x_6}$.

#### Step $i = 3$
The available variable sets to choose between is obtained by taking the current variable set ${x_1, x_6}$ and adding each of the left-out variables thereby resulting in the sets ${x_1, x_6, x_7}$, ${x_1, x_7, x_8}$, ${x_1, x_6, x_8}$, ${x_6, x_7, x_8}$, ${x_1, x_9}$, ${x_6, x_7, x_9}$, ${x_1, x_8, x_9}$, ${x_7, x_8, x_9}$. Since the lowest error of the available sets is 4.161, which is lower than 4.213, we update the current selected variables to ${x_1, x_6, x_8}$.

#### Step $i = 4$
The available variable sets to choose between is obtained by taking the current variable set ${x_1, x_6, x_8}$ and adding each of the left-out variables thereby resulting in the sets ${x_1, x_6, x_7, x_8}$, ${x_1, x_6, x_7, x_9}$, ${x_1, x_6, x_8, x_9}$, ${x_1, x_7, x_8, x_9}$, ${x_6, x_7, x_8, x_9}$. Since the lowest error of the available sets is 4.098, which is lower than 4.161, we update the current selected variables to ${x_1, x_6, x_7, x_8}$.

#### Step $i = 5$
The available variable sets to choose between is obtained by taking the current variable set ${x_1, x_6, x_7, x_8}$ and adding each of the left-out variables thereby resulting in the sets ${x_1, x_6, x_7, x_8, x_9}$. Since the lowest error of the newly constructed sets is not lower than the current error the algorithm terminates.

### Backward selection:
The method is initialized with the set ${x_1, x_6, x_7, x_8, x_9}$ having an error of 4.195.

#### Step $i = 1$
The available variable sets to choose between is obtained by taking the current variable set ${x_1, x_6, x_7, x_8, x_9}$ and removing each of the left-out variables thereby resulting in the sets ${x_1, x_6, x_7, x_8}$, ${x_1, x_6, x_7, x_9}$, ${x_1, x_6, x_8, x_9}$, ${x_1, x_7, x_8, x_9}$, ${x_6, x_7, x_8, x_9}$. Since the lowest error of the available sets is 4.098, which is lower than 4.195, we update the current selected variables to ${x_1, x_6, x_7, x_8}$.

#### Step $i = 2$
The available variable sets to choose between is obtained by taking the current variable set ${x_1, x_6, x_7, x_8}$ and removing each of the left-out variables thereby resulting in the sets ${x_1, x_6, x_7}$, ${x_1, x_6, x_8}$, ${x_1, x_7, x_8}$, ${x_6, x_7, x_8}$. Since the lowest error of the newly constructed sets is not lower than the current error, the algorithm terminates.
