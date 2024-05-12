

### Linear Regression Basics

**Linear regression** aims to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. The equation of a linear model can be expressed in the general form:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon
$$

where:
- $y$ is the dependent variable,
- $(x_1, x_2, \dots, x_n)$ are the independent variables,
- $(\beta_0, \beta_1, \dots, \beta_n)$ are the parameters of the model,
- $(\epsilon)$ is the error term, which accounts for the variability in $y$ not explained by the linear relationship.

#### Key Concepts:

1. **Parameter Estimation**: Parameters are estimated using the method of least squares, which minimizes the sum of the squared differences between observed values and values predicted by the model.
2. **Feature Transformation**: It allows linear models to fit non-linear data by applying transformations to the input features, like polynomial and trigonometric transformations.

#### Example:
Imagine you're predicting house prices based on size (in square feet). A simple linear model could be:

$$
\text{Price} = \beta_0 + \beta_1 \times \text{Size}
$$

If $(\beta_0 = 150,000)$ and $(\beta_1 = 100)$, then for a house of 1000 square feet, the predicted price would be:

$$
150,000 + 100 \times 1000 = \$250,000
$$



**Logistic Regression**:
- **Purpose**: To predict binary outcomes (0 or 1) based on one or more predictors.
- **Model**: The probability of the dependent variable equaling a class (e.g., 1) is modeled using the logistic function:
  $$
  P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n)}}
  $$

#### **Training Linear Models**

- **Least Squares for Linear Regression**: This method minimizes the sum of the squared differences between observed and predicted values.
- **Maximum Likelihood for Logistic Regression**: This approach estimates parameters that maximize the likelihood of the observed sequence of labels given the training data.

#### **Feature Transformation**

- **Polynomial Regression**: Extends linear models by adding polynomial terms, which allows the model to fit non-linear relationships.
  $$
  y = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_nx^n
  $$
- **Interaction Terms**: Used to capture the interaction effects between variables.

####  **Assumptions of Linear Models**

- **Linearity**: The relationship between dependent and independent variables is linear.
- **Homoscedasticity**: Constant variance of error terms.
- **Independence**: Observations are independent of each other.
- **Normality**: Residuals (errors) of the model are normally distributed.

#### **Evaluating Model Fit**

- **R-squared and Adjusted R-squared**: Measures used to assess the fit of a regression model, indicating the proportion of variance explained by the model.
- **Confusion Matrix for Classification**: A table used to describe the performance of a classification model, showing the true positives, false positives, true negatives, and false negatives.

#### Practical Examples

**Example for Linear Regression**:
Suppose you're modeling house prices (in thousands) as a function of size (square feet). A simple model could be:
$$
\text{Price} = 50 + 0.2 \times \text{Size}
$$
If a house is 2000 square feet, the predicted price would be:
$$
\text{Price} = 50 + 0.2 \times 2000 = 450 \text{ thousand dollars}
$$

**Example for Logistic Regression**:
Predicting whether a student passes (1) or fails (0) based on hours studied. Given model parameters $\beta_0 = -4$ and $\beta_1 = 0.5$, the probability of passing for a student who studies 10 hours is:
$$
P(\text{Pass}) = \frac{1}{1 + e^{-(-4 + 0.5 \times 10)}} \approx 0.88
$$

### Problems from Chapter 8

#### Example Problem:
- **Linear Regression Problem**: Predicting the fuel efficiency of cars based on their engine size and weight.
- **Logistic Regression Problem**: Classifying emails as spam or not based on the frequency of certain keywords and the presence of attachments.

### Problem 8.1 & 8.2 Explanation 

**Problem 8.1:**
- The scenario involves a model where the wind direction and the hour of the day are not considered in the prediction of pollution levels, which seems reasonable as they do not directly influence pollution in a linear manner. Other variables like the number of cars and wind speed have direct coefficients showing their impact.
![[Pasted image 20240510142357.png]]

**Problem 8.2:**
- This problem involves a logistic regression model predicting whether a consumer is from Lisbon or Oporto based on various standardized attributes. The model indicates that factors like fresh food and paper consumption, despite their small coefficients, might still influence predictions, underscoring the importance of all features in the model.
![[Pasted image 20240510142450.png]]
![[Pasted image 20240510142440.png]]

Let's now solve these two specific problems using provided details:

### Solving Problems 8.1 & 8.2

#### Problem 8.1 - Model Evaluation:
The given linear regression model factors indicate:
- **WINDDIR and HOUR**: Not included in the model, thus irrelevant.
- **LogCAR**: Positive coefficient suggests more cars increase pollution.
- **WIND**: Negative coefficient suggests higher wind speeds might help disperse pollution, thus reducing levels.
- **TEMP**: Negative coefficient indicates that higher temperatures might reduce pollution levels, possibly due to different atmospheric conditions affecting chemical reactions in the air.

#### Problem 8.2 - Logistic Regression Insights:
The model uses standardized inputs to predict geographic origin based on consumer behavior. Notable insights:
- Negative coefficients for attributes $(x_1)$, $(x_2)$, and $(x_6)$ suggest that higher values of these attributes are associated with a higher likelihood of the consumer being from Lisbon.


### Bernoulli Distribution 

The **Bernoulli distribution** is a discrete distribution having two possible outcomes labeled by $0$ and $1$, where $1$ usually represents "success" and $0$ represents "failure". It is particularly useful in scenarios where there are only two possible outcomes of a random experiment or process.

#### Definition
A random variable $X$ follows a Bernoulli distribution if its probability mass function (PMF) is given by:
$$
P(X = x) = p^x(1-p)^{1-x} \quad \text{for } x = 0, 1
$$
where:
- $p$ is the probability of success ($X=1$),
- $1-p$ is the probability of failure ($X=0$).

#### Properties
- **Mean (Expected value)**: The mean of a Bernoulli random variable $X$ is $p$.
- **Variance**: The variance of $X$ is $p(1-p)$.

#### Example
Consider a fair coin toss where the outcome of getting "heads" is a success ($X=1$) and "tails" is a failure ($X=0$). If the coin is fair, $p = 0.5$. The PMF can be described as:
$$
P(X=1) = 0.5 \quad \text{and} \quad P(X=0) = 0.5
$$
The mean and variance of this Bernoulli distribution would be $0.5$ and $0.25$, respectively.

### Confusion Matrix Explained

A **confusion matrix** is a summary table used in classification problems. It describes the performance of a classification model by comparing the actual outcomes with the predicted outcomes from the model. It helps identify how well the model is distinguishing between different classes.

#### Structure
The confusion matrix for a binary classifier includes:
- **True Positives (TP)**: Correctly predicted positive observations.
- **True Negatives (TN)**: Correctly predicted negative observations.
- **False Positives (FP)**: Incorrectly predicted as positive (Type I error).
- **False Negatives (FN)**: Incorrectly predicted as negative (Type II error).

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | True Positives (TP)  | False Negatives (FN) |
| **Actual Negative** | False Positives (FP) | True Negatives (TN) |

#### Example
Imagine a scenario where you have developed a model to predict whether an email is spam (1) or not spam (0). After testing the model on 100 emails, you get the following results:
- 40 emails are correctly identified as spam (TP).
- 45 emails are correctly identified as not spam (TN).
- 5 emails are not spam but wrongly identified as spam (FP).
- 10 emails are spam but wrongly identified as not spam (FN).

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | 40 (TP)             | 10 (FN)             |
| **Actual Negative** | 5 (FP)              | 45 (TN)             |


From this matrix, you can compute various performance metrics such as accuracy, precision, recall, and F1-score:
- **Accuracy** (Overall effectiveness of the classifier):
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{40 + 45}{100} = 0.85
  $$
- **Precision** (Accuracy of positive predictions):
  $$
  \text{Precision} = \frac{TP}{TP + FP} = \frac{40}{40 + 5} = 0.89
  $$
- **Recall** (Coverage of actual positive sample):
  $$
  \text{Recall} = \frac{TP}{TP + FN} = \frac{40}{40 + 10} = 0.80
  $$

These metrics derived from the confusion matrix help in understanding the strengths and weaknesses of the classification model, particularly in how different types of errors are being made.

