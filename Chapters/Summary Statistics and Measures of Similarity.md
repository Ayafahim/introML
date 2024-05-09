

#### 4.1 Attribute Statistics

**Empirical (hat symbol ^)** signifies “best guess based on the available information” according to the book.

We have formulas for Empirical Mean, Variance, and Standard Deviation of $x$:

![[Pasted image 20240509213951.png]]

##### Empirical Mean:
Calculates the mean by summing up all individual values in the dataset and then dividing that total by the number of observations (provides average value of the dataset).

###### Components of the Mean Formula:
- $\mu \sim$ is the mean or average meaning 50% in a dataset.
- $N$ is the number of observations in a dataset.
- $X_i$ represents each individual observation, like $X_1$ could be 2, $X_2 = 5$, etc.
- $\Sigma$ (Sigma) indicates summation over the indexed values. (Sum of all values in a dataset)

##### Empirical Variance:
Provides a measure of the variability or dispersion of a set of data points around their mean value.

###### Components of the Variance Formula:
- $s$ represents a measure of the spread or dispersion of a dataset around its mean.
- $\mu$ is the mean, as you see the formula uses the mean value which takes the average value of a dataset.
- $N$ is the number of observations.
- $\sum$: Summation symbol, indicating that the following expression is to be summed over all $i$ from 1 to $N$.

###### Steps to Calculate the Variance:
1. Calculate mean.
2. Calculate the difference from mean and $x_i - \mu$, meaning $x_1 - \mu$, $x_2 - \mu$ etc.
3. Square the Differences: Square each of these differences. (eliminates negative values).
4. Sum of Squared Differences.
5. Normalize. Divide this total by $N-1$ (not $N$ because using $N-1$ provides an unbiased estimator under the assumption that the data are sampled from a larger population, known as Bessel's correction).

##### Standard Deviation:
Standard deviation is essentially the square root of the average squared deviations from the mean. It tells you, on average, how much each data point differs from the mean of the dataset.

###### Components of the SD Formula:
- $\sigma$ = Taking the square root of the variance computes the standard deviation.

##### Example of how to calculate Empirical Mean, SD, and Variance:
I will take account for Q24 in 21Fall exams. We are given the following dataset:

- Data = [1, 3, 3, 1, 2, 3, 1]
- $N = 7$, we have 7 observations
- $X_i = x_1 + \ldots x_7$
- We take the sum of the $X_i$: $X_i = 1 + 3 + 3 + 1 + 2 + 3 + 1 = 14$
- $\mu = \frac{14}{7} = 2$

The empirical mean is 2.

Now we will calculate the empirical SD:
$$
\hat{s} \approx \frac{1}{N-1} \sum_{i=1}^N (x_i - \hat{\mu})^2
$$

- We will start with calculate the difference in mean so: meaning $x_i - \mu$
- $x_1 - \mu$, $x_2 - \mu$, $x_3 - \mu \ldots x_7 - \mu$
- [1−2, 3−2, 3−2, 1−2, 2−2, 3−2, 1−2]=[−1, 1, 1, −1, 0, 1, −1]
- Squared differences (eliminates negative values)
- Takes the sum of the squared differences:
- $1 + 1 + 1 + 1 + 0 + 1 = 6$
- Then we calculate empirical variance
- $s^2 = \frac{6}{(7 - 1)} = 1$

Thereby the empirical variance is 1.

We will now calculate the empirical SD:
$$
\hat{\sigma} = \sqrt{\hat{s}}
$$ 

 $$
 \sigma = \sqrt{1} = 1
 $$

This is how you calculate the empirical mean, standard deviation, and variance.

##### Median Calculation:
We will again use the following dataset:
- Data = [1, 3, 3, 1, 2, 3, 1]
- If the number of observations $N$ is odd, the median is the middle number once all values are ordered.
- If $N$ is even, the median is the average of the two middle numbers.
- Remember the data get sorted.
- Mean = $\frac{X_i}{N} = \frac{1 + 3 + 3 + 1 + 2 + 3 + 1}{7} = 2$

#### Percentile Calculation

Percentiles are a statistical measure that indicates the value below which a given percentage of observations in a group of observations fall. For example, the 90th percentile is the value below which 90% of the observations may be found.

##### How to Calculate Percentiles:
1. **Sort Your Data**: Arrange all your data points in order from smallest to largest.
2. **Find the Position**: To determine which value represents a specific percentile, use the formula:
   
   $$P = \left(\frac{n + 1}{100}\right) \times p$$

   where:
   - $P$ is the position of the percentile in the data set.
   - $n$ is the total number of observations in the dataset.
   - $p$ is the desired percentile (e.g., 90 for the 90th percentile).

3. **Interpolation (if necessary)**: If $P$ does not correspond to a whole number, interpolate between the two surrounding data points to estimate the percentile value.

This method allows you to identify various percentile levels within a dataset, providing insights into the distribution of the data, such as identifying the median (50th percentile), quartiles (25th, 50th, 75th percentiles), or any other positional statistics.

### Example:

Imagine you have the grades of 10 students: [55, 60, 65, 70, 75, 80, 85, 90, 95, 100], and you want to find the 80th percentile.

- **Sort the data:** Already sorted.
- **Find the position:**

  $$\text{Position} = \left(\frac{10 \times 80}{100}\right) + 0.5 = 8.5$$

- **Get the 80th percentile value:** The 8th number is 90 and the 9th number is 95, so:

  $$80\text{th percentile} = \frac{90 + 95}{2} = 92.5$$

So, 80% of the grades are below 92.5. This means if you got a 92.5, you scored better than 80% of the students.


#### Covariance and Correlation

##### Covariance
Covariance measures how two variables (say, $x$ and $y$) move together. If they tend to increase and decrease together, the covariance is positive. If one tends to increase when the other decreases, the covariance is negative. Remember that all values inside a covariance matrix are variance values for $x_i$ and $y_i$.

![[Pasted image 20240509215255.png]]

##### Correlation
Correlation, like covariance, measures the relationship between two variables, but it gives you a standardised score between -1 and 1. This score tells you not only the direction of the relationship (like covariance) but also its strength.

- A correlation of 1 means there's a perfect positive relationship (as one variable increases, so does the other).
- A correlation of -1 means there's a perfect negative relationship (as one variable increases, the other decreases).
- A correlation of 0 means no linear relationship exists between the variables.

![[Pasted image 20240509215452.png]]

### Formula for Correlation

Correlation is calculated as the ratio of the covariance of two variables to the product of their standard deviations:

$$
\text{Correlation} = \frac{\text{Covariance}}{\sigma_x \sigma_y} = \frac{\frac{1}{N-1} \sum_{i=1}^N (x_i - \mu_x)(y_i - \mu_y)}{\sigma_x \sigma_y}
$$

Where:
- $\sigma_x$ and $\sigma_y$ are the standard deviations of $x$ and $y$ respectively.

###### Example of How to Calculate Correlation from F19: Q5

![[Pasted image 20240509215804.png]]
![[Pasted image 20240509215907.png]]


- Month is $x_1$ while $PM_{2.5}$ is $x_2$.
- We can therefore see that our $Cov(x_1, x_2) = -29$.
- We will use the following formula for empirical correlation:

Since we already know the Covariance which is $-29$ and we know that the variance for $x_1$ and $x_2$ is 12 and 6014 which is $Cor(x_1, x_2)$. Since we know what the empirical variance is we can use the formula for sd which is $\sqrt{x_1 * x_2}$ .

  
  $$\text{Correlation}(x_1, x_2) = \frac{-29}{\sqrt{12} \times \sqrt{6104}} = \frac{-29}{270.6} \approx -0.1072$$

```python
import numpy as np # type: ignore

# Given empirical covariance matrix
covariance_matrix = np.array([
    [12, -29, -21, -12, -317],
    [-29, 6104, 6026, 1557, 67964],
    [-21, 6026, 7263, 1701, 70892],
    [-12, 1557, 1701, 1012, 25415],
    [-317, 67964, 70892, 25415, 1212707]
])

# Extract the covariance between MONTH (x1) and PM2.5 (x2) --> Change this according to the question
cov_x1_x2 = covariance_matrix[0, 1]

# Variance of MONTH (x1) and PM2.5 (x2)
var_x1 = covariance_matrix[0, 0]
var_x2 = covariance_matrix[1, 1]

# Calculate the empirical correlation coefficient
correlation_x1_x2 = cov_x1_x2 / np.sqrt(var_x1 * var_x2)
print(correlation_x1_x2)
```


#### 4.2 Term-Document Matrix
Imagine you have several stories written down, and you want to see which words are used in each story. A term-document matrix is like a big chart where:
- Each row represents one of the stories.
- Each column represents a specific word.
- The numbers in the chart show how many times each word appears in each story.
![[Pasted image 20240509220444.png]]


#### 4.3 Measures of Distance

##### Distance
When we talk about distance in machine learning, we're often trying to figure out how different or similar two things are. For example, if you have pictures of cats and dogs, a machine learning program can use "distance" to decide whether a new picture looks more like the cats or the dogs it has seen before.

###### Basic Rule of Distance
- **Non-negativity**: Distance can never be negative; it's always zero or more.
- **Identity of indiscernibles**: If the distance between two items is zero, they must be the same thing.
- **Symmetry**: The distance from one item to another is the same in either direction.
- **Triangle inequality**: This is like saying if you walked from your house to school and then to the park, this should be longer than going straight from your house to the park.
![[Pasted image 20240509220602.png]]

##### Norm
A norm is a way of measuring the size or length of something in mathematics, especially when dealing with numbers or coordinates. You can think of it like measuring how far a point is from the center of a map.

The most common type of norm used to measure distance is called the Euclidean norm. It's just like measuring a straight line between two points.

###### Example:
Imagine you have a point at $(3,4)$ on a graph. The Euclidean norm measures how far this point is from the origin $(0,0)$ using the formula:

$$\text{Distance} = \sqrt{3^2 + 4^2} = 5$$

To measure the distance between two different points, say $(3,4)$ and $(1,1)$, subtract one fromthe other and then apply the norm:
- Subtract the coordinates: $(3-1, 4-1) = (2,3)$
- Then calculate: $\sqrt{2^2 + 3^2} = \sqrt{13}$

These concepts help computers and programs understand and process complex data like images, sounds, or patterns of behavior by turning them into numbers and measuring how these numbers are similar or different. This helps in tasks like recognizing what's in a photo or suggesting songs that sound similar.

#### P-Norms
P-Norms are a way to measure distances, where "p" can change how we calculate this measurement.

###### Types of P-Norms:
- When $p=1$: This is like adding up all the differences block by block (or point by point). It's straightforward - just add up how much each point differs.
- When $p=2$ (Euclidean Norm): This is the most common way to measure. It's like drawing a straight line between two points and measuring that line.
- When $p$ is very large (approaching infinity): This means we only care about the single biggest difference. Imagine you have several blocks, and you only care about the longest one.
- When $p<1$: These are less common and a bit tricky because they don't behave like usual measurements (they can behave oddly when you combine or compare them).

###### General P-Norm Calculation:
Raise each difference to the power of $p$, add them all up, and then take the $p$-th root (like an average but adjusted by $p$).

#### Frobenius Norm
When dealing with matrices (grids of numbers), sometimes we need to measure the "size" of the entire grid. The Frobenius norm helps us do that. It's like measuring every block in a large structure and combining those measurements into one overall size.
$$
\text{Frobenius Norm} = \sqrt{\text{sum of the squares of each element in the matrix}}
$$

![[Pasted image 20240509220802.png]]
#### Mahalanobis Distance
The Mahalanobis distance is a measure of distance between a point and a distribution. It is an effective way of determining similarity of an unknown sample set to a known one.

#### Similarity Measures

##### SMC (Simple Matching Coefficient)
The SMC calculates the proportion of positions where the two vectors match both in terms of having 1s and having 0s.
$$\text{SMC}(x, y) = \frac{f_{11} + f_{00}}{M}$$

##### Jaccard Similarity
Jaccard similarity is a statistic used to gauge the similarity and diversity of sample sets.
$$\text{Jaccard Similarity}(x, y) = \frac{f_{11}}{f_{11} + f_{10} + f_{01}}$$

##### Cosine Similarity
Cosine similarity, a commonly used measure in fields such as data science and text analysis, calculates how similar two sets of data (often vectors) are.
$$\cos(x, y) = \frac{f_{11}}{\|x\| \|y\|}$$

###### Example of Cosine Similarity and Term-Document Matrix from 18F Q11:
In our $n_1=12$, $n_2=7$ based on how many words $s_1$ and $s_2$ contain. Of those 19 words, 4 of the words are unique which are “the, representation, should and a” since they occur in both $s_1$ and $s_2$ meaning that's our $f_{11} = 4$.

If we use the cosine similarity formula:
$$\cos(x, y) = \frac{f_{11}}{n_1 \times n_2} = \frac{4}{12 \times 7} = 0.044$$

Script for the task: Q11_cosine-similarity-Term-Document-Matrix
