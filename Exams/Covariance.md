## S19, Q26) What is the correlation between variables x1 and x2?
![[Pasted image 20240516025108.png]]
![[Pasted image 20240516025032.png]]

First you can clearly see that it is the first matrix since the plot for x1,x2 should have a positive diagonal therefore it is the first matrix. Now the correlation between x1 and x2 is:
$$
corr(x_1,x_2) = \frac{cov(x_1,x_2)}{\sqrt{var(x_1) \times var(x_2)}}
$$
Where $cov(x_1,x_2)$ = 0.56 & the variances are in the diagonal so $var(x_1) = 0.5, var(x_2) = 1.5$
$$
corr(x_1,x_2) = \frac{0.56}{0.5 \times 1.5 } = 0.647
$$
---
## S23, Q2) Calculate covariance 

![[Pasted image 20240520202225.png]]

---
## F23, Q1) Which one of the following matrices represents the empirical correlation matrix for these attributes?
![[Pasted image 20240517225932.png]]
![[Pasted image 20240517225927.png]]

### Solution

If you look at option b you can see $cov(x_1,x_4)$ is 0.55 which doesnt match since we can see it has a negative tilt which means it should be negative.

It cant be option D because 0.25 is a very small covariance for $x_1,x_2$ since they are strongly correlated. And for option A we see $x_3,x_4$ are not correlated so 0.72 is too high, it should be closer to 0, therefore its option C.


---
## F16, Q22)

![[Pasted image 20240521152851.png]]

![[Pasted image 20240521152915.png]]