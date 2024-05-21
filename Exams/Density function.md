

## S20, Q24) In Figure 10 is given the denstity function p(x) of a random variable x. What is the expected value of x, i.e. $E[x]$?
![[Pasted image 20240521202120.png]]


Look at the intervals e.g. when x is between 0 to 0.2 : $x=0  ; 0.2$ then $p(x)=0.6$ 
when $x=0.2;0.6$, $p(x)=1$ and when $x=0.6;0.9$ then $p(x) = 1.6$

To get the area under we calculate the integrals, i did it in maple:

$$
\int_{0}^{0.2} 0.6x \, dx + \int_{0.2}^{0.6} x \, dx + \int_{0.6}^{0.9} 1.6x \, dx = 0.5320000000
$$
