## F23, Q12) standardize the input before learning the optimal bias  $w^∗_0$ and parameter $w^*_1$ for a regularized linear regression model with λ = 0.7, where we consider the formulation from Section 14.1. What is the prediction of the trained model for the input $x_2$?

### Solution

### Given Information
- Input values: $([-3.4, -1.3, 0.5, 2.4, 4.2])$
- Output values: $([-2.9, -0.4, 0.7, 2.5, 4.5])$
- Regularization parameter: $(\lambda = 0.7)$
- We are standardizing the input before fitting the model.

### Steps in the Solution

1. **Standardize the Input:**
   - Compute the mean $(\mu)$ and standard deviation $(\sigma)$ of $(x)$:

     $$\mu = \frac{-3.4 + (-1.3) + 0.5 + 2.4 + 4.2}{5} = 0.48$$

     $$\sigma \approx 2.9895$$

   - Standardize \( x \):

     $$\tilde{x}_i = \frac{x_i - \mu}{\sigma}$$

2. **Formulate the Regularized Linear Regression:**
   - Use the formula for finding the optimal parameters:

     $$w_0^* = \frac{1}{5} \sum y_i = \frac{-2.9 + (-0.4) + 0.7 + 2.5 + 4.5}{5} = 0.88$$

     $$w_1^* = \frac{\sum \tilde{x}_i (y_i - w_0^*)}{\sum \tilde{x}_i^2 + \lambda}$$

3. **Calculate $(\tilde{x}_i)$ values:**
   - Standardized values:

     $$\tilde{x}_1 = \frac{-3.4 - 0.48}{2.9895} \approx -1.302$$

     $$\tilde{x}_2 = \frac{-1.3 - 0.48}{2.9895} \approx -0.595$$

     $$\tilde{x}_3 = \frac{0.5 - 0.48}{2.9895} \approx 0.007$$

     $$\tilde{x}_4 = \frac{2.4 - 0.48}{2.9895} \approx 0.643$$

     $$\tilde{x}_5 = \frac{4.2 - 0.48}{2.9895} \approx 1.244$$

4. **Calculate $( w_1^* )$:**
   - Sum up the numerator and denominator separately:

     $$\text{Numerator} = \sum \tilde{x}_i (y_i - w_0^*)$$
     $$= (-1.302 \cdot (-2.9 - 0.88)) + (-0.595 \cdot (-0.4 - 0.88)) + (0.007 \cdot (0.7 - 0.88)) + (0.643 \cdot (2.5 - 0.88)) + (1.244 \cdot (4.5 - 0.88))$$
     $$= (1.302 \cdot 3.78) + (0.595 \cdot 1.28) + (0.007 \cdot -0.18) + (0.643 \cdot 1.62) + (1.244 \cdot 3.62)$$
     $$= 4.92 + 0.76 - 0.001 + 1.04 + 4.5 \approx 11.22$$

     $$\text{Denominator} = \sum \tilde{x}_i^2 + \lambda$$
     $$= (-1.302)^2 + (-0.595)^2 + (0.007)^2 + (0.643)^2 + (1.244)^2 + 0.7$$
     $$= 1.695 + 0.354 + 0.00005 + 0.413 + 1.548 + 0.7 \approx 4.71$$

   - Calculate $( w_1^* )$:

     $$w_1^* = \frac{11.22}{4.71} \approx 2.38$$

5. **Prediction for $( x_2$):**
   - Standardized value of $( x_2)$:

     $$\tilde{x}_2 = -0.595$$

   - Prediction using the trained model:

     $$\hat{y}_2 = w_0^* + w_1^* \cdot \tilde{x}_2 = 0.88 + 2.38 \cdot (-0.595) = 0.88 - 1.415 = -0.535$$

### Conclusion
The correct answer is:
B. $( \hat{y} \approx -0.54)$

### Summary Notes:
1. **Standardization:**
   - Subtract the mean and divide by the standard deviation.

2. **Regularized Linear Regression:**
   - Use the formula $( w_0^* = \frac{1}{5} \sum y_i )$
   - Use $( w_1^* = \frac{\sum \tilde{x}_i (y_i - w_0^*)}{\sum \tilde{x}_i^2 + \lambda})$

### Script
```python 
import numpy as np  
  
  
# Synthetic dataset  
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])  
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])  
  
# Standardization parameters  
mu = np.mean(x)  
sigma = np.std(x, ddof=1)  # Using ddof=1 for sample standard deviation  
  
# Standardize x  
x_standardized = (x - mu) / sigma  
  
# Lambda for regularization  
lambda_ = 0.7  
  
# Calculate the intercept w0 as per the provided formula  
w0 = np.mean(y)  # Since the formula was w0 = 1/5 * sum(yi), which simplifies to the mean of y  
  
# Calculate w1 using the corrected formula  
w1 = np.sum(x_standardized * (y - w0)) / (np.sum(x_standardized**2) + lambda_)  
  
# Predicting for x2 (value at index 1 in the original x array, corresponding to -1.3)  
x2_standardized = (-1.3 - mu) / sigma # This needs to be looked in the table to make sure it is the right number  
y_pred = w1 * x2_standardized + w0  
  
print("w0:", w0)  
print("w1:", w1)  
print("Predicted y for x2:", y_pred)
```
