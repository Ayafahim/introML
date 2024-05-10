## Chapter 8: Linear Models Overview

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

