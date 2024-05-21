

## S22, Q10)



To solve Question 10, we need to modify the logistic regression cost function for an alternative encoding of the class label. Here’s a step-by-step solution:

### Question 10 Explanation

The standard logistic regression cost function for a binary classification problem is given by:

$$ E(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(x_i^T w)) + (1 - y_i) \log(1 - \sigma(x_i^T w)) \right] $$

where $ \sigma(z) = \frac{1}{1 + e^{-z}} $ is the logistic function.

In the alternative encoding:
- $ y_i $ is encoded as $ y_i = -1 $ for the negative class and $ y_i = +1 $ for the positive class.

We need to adjust the cost function $ E(w) $ to work with this encoding. The alternative cost function $ \tilde{E}(w) $ should maintain the same logistic regression functionality.

### Alternative Encoding Cost Function Derivation

The logistic function can be rewritten for the alternative encoding. For $ y_i \in \{-1, +1\} $:

$$ \sigma(x_i^T w) = \frac{1}{1 + e^{-x_i^T w}} $$

In the new encoding, we want $ y_i \sigma(x_i^T w) $ to reflect the probability of the correct class:

$$ \sigma(y_i x_i^T w) = \frac{1}{1 + e^{-y_i x_i^T w}} $$

Hence, the cost function $ \tilde{E}(w) $ becomes:

$$ \tilde{E}(w) = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{1}{1 + e^{-y_i x_i^T w}} \right) $$

This simplifies to:

$$ \tilde{E}(w) = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \sigma(y_i x_i^T w) \right) $$

By recognizing that $ \log(\sigma(z)) = -\log(1 + e^{-z}) $, we get:

$$ \tilde{E}(w) = \frac{1}{N} \sum_{i=1}^{N} \log(1 + e^{-y_i x_i^T w}) $$

Given this, the correct cost function from the provided options must correspond to this derived form.

### Solving the Question

Looking at the provided options:

- **Option A**: $\tilde{E}(w) = - \frac{1}{N} \log \left( \prod_{i=1}^{N} y_i \frac{1}{1+e^{-x_i^T w}} \right)$
- **Option B**: $tilde{E}(w) = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \left( \frac{1}{1+e^{-x_i^T w}} \right)^{y_i} \right)$
- **Option C**: $\tilde{E}(w) = - \frac{1}{N} \sum_{i=1}^{N} \log \left( y_i \frac{1}{1+e^{-x_i^T w}} \right)$
- **Option D**: $\tilde{E}(w) = - \frac{1}{N} \log \left( \prod_{i=1}^{N} \frac{1}{1+e^{-y_i x_i^T w}} \right)$

We are looking for a function that minimizes the log-likelihood of the alternative encoding.

By examining the options:
- **Option D**: Aligns perfectly with our derived form:

$$ \tilde{E}(w) = - \frac{1}{N} \log \left( \prod_{i=1}^{N} \frac{1}{1 + e^{-y_i x_i^T w}} \right) $$

Thus, the correct answer is:

$$ \boxed{D} $$
