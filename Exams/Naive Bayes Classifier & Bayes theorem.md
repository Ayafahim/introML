## S19, Q13) what is then the probability it has average rating (y = 2) according to the Naıve-Bayes classifier?
![[Pasted image 20240515231345.png]]

![[Pasted image 20240515231403.png]]

### Solution
Remember that:

$$
p_{NB}(y = 2 \mid f_2 = 0, f_4 = 1, f_5 = 0) = \frac{p(f_2 = 0 \mid y = 2) p(f_4 = 1 \mid y = 2) p(f_5 = 0 \mid y = 2) p(y = 2)}{\sum_{j=1}^3 p(f_2 = 0 \mid y = j) p(f_4 = 1 \mid y = j) p(f_5 = 0 \mid y = j) p(y = j)}
$$
This:
$$
{\sum_{j=1}^3 p(f_2 = 0 \mid y = j) p(f_4 = 1 \mid y = j) p(f_5 = 0 \mid y = j) p(y = j)}
$$
Means to multiply all times where f2=0 , f4 = 1 and f5 = 0 and then also amount of classes for y  for all the classes and sum them.
So fx:


$$
= \frac{\frac{2 \times 2 \times 2 \times 3}{3 \times 3 \times 3 \times 10}}{\frac{2 \times 1 \times 2 \times 2}{2 \times 2 \times 2 \times 10} + \frac{2 \times 2 \times 2\times 3}{3 \times 3\times 3 \times 10} + \frac{4 \times 3 \times 1\times 5}{5 \times 5\times 5 \times 10}} =  \frac{533}{200}
$$
 
${p(f_2 = 0 \mid y = 2)} = 2/3$ times if we look at the table 
${p(f_4 = 1 \mid y = 2)} = 2/3$ times if we look at the table 
${(y = 2)} = 3/10$ because there are 3 classes where y = 2

This is done for all of them.

calculated in maple
```maple
2/3*2/3*2/3*3/10/(2/2*1/2*2/2*2/10 + 2/3*2/3*2/3*3/10 + 4/5*3/5*1/5*5/10)
```


## S19, Q20) Using this, what is then the probability an observation had poor rating given that ˆx2 = 0 and ˆx3 = 1?

![[Pasted image 20240516030011.png]]
![[Pasted image 20240516030016.png]]

### Solution

Remember that:

$$
p(y = 1 \mid x_2 = 0, x_3 = 1) = \frac{p(x_2 = 0, x_3 = 1 \mid y = 1)p(y = 1) }{\sum_{j=1}^3 p(x_2 = 0 \mid y = j) p(x_3 = 1 \mid y = j)p(y = j)}
$$

So:

$$
p(y = 1 \mid x_2 = 0, x_3 = 1) = \frac{0.17 \times 0.268}{0.28 \times 0.366 + 0.17\times 0.268 + 0.33 \times 0.365} = 0.17
$$

---
## F31, Q11) Consider the Table 4 that shows the class conditional joint probability for the attributes $x_1, x_3$ of the CCPP dataset after binarization, while the prior probabilities for the two classes are $p(y =  Low) = 0.53$ and $p(y = High) = 0.47$.  What is the probability the energy production to be High when $x_1 = 0$?

![[Pasted image 20240518004150.png]]

### Solution 
You just have to first take $p(x_1 = 0 | y = high)$  so this means for both $p(x_1 = 0 | x_3 = 0)$ and $p(x_1 = 0 | x_3 = 1)$

so:
$$
p(x_1 = 0 | y = high) = p(x_1 = 0 | x_3 = 1) + p(x_1 = 0 | x_3 = 0)
$$ $$
p(x_1 = 0 | y = high) = 0.25 + 0.68
$$
and for y = low:
$$
p(x_1 = 0 | y = low) = p(x_1 = 0 | x_3 = 1) + p(x_1 = 0 | x_3 = 0)
$$
so:
$$
\frac{0.47 \times (0.25 + 0.68)} {(0.47 \times (0.25 + 0.68) + 0.53 \times (0.04 + 0.03))} = 0.92
$$
#### Solution from exam pdf
![[Pasted image 20240518004947.png]]
