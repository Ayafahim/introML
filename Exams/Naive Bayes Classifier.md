## S19, Q13) what is then the probability it has average rating (y = 2) according to the Na ̈ıve-Bayes classifier?
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
