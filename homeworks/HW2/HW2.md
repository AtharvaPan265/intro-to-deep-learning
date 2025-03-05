# Problem 1
Use back-propagation to calculate the gradients of $$f(W,x)=||\sigma(Wx)||^2$$
with respect to x and W. Here, $∥\cdot∥^2$ is the calculation of L2 loss, $W$ is a $3×3$ matrix, and $x$ is a $3 × 1$ vector, and $\sigma(\cdot)$ is the ReLU function that performs element-wise operation.

We can first write out $W$ and $x$

$$
W = 
\begin{bmatrix}
W_{1,1} & W_{1,2} & W_{1,3}\\
W_{2,1} & W_{2,2} & W_{2,3}\\
W_{3,1} & W_{3,2} & W_{3,3}
\end{bmatrix},\ \ x = \begin{bmatrix} x_{1}\\ x_{2}\\ x_{3} \end{bmatrix}
$$
Let's say:
$$
z = \begin{bmatrix}
W_{1,1}x_1 + W_{1,2}x_2 + W_{1,3}x_3\\
W_{2,1}x_1 + W_{2,2}x_2 + W_{2,3}x_3\\
W_{3,1}x_1 + W_{3,2}x_2 + W_{3,3}x_3
\end{bmatrix}
=
\begin{bmatrix}
z_{1}\\
z_{2}\\
z_{3}
\end{bmatrix}

$$
so to do $a=\sigma(z)$
$$
a = 
\begin{bmatrix}
max(0,z_1)\\
max(0,z_2)\\
max(0,z_3)
\end{bmatrix} =
\begin{bmatrix}
a_{1}\\
a_{2}\\
a_{3}
\end{bmatrix}  =
\begin{cases} z_i & z_i > 0 \\ 0 & z_i \leq 0 \end{cases} $$


Now we are left with $$f(W,x)=||a||^2$$
then Gradient with respect to $\mathbf{a}$ 
$$\dfrac{\partial f}{\partial a} = \dfrac{\partial }{\partial a}(a_1^2 + a_2^2 + a_3^2) = 2a = 
\begin{bmatrix}
2a_{1}\\
2a_{2}\\
2a_{3}
\end{bmatrix}$$
so we get the gradient of $f$ with respect to $a$ $$\nabla_af=2a$$
then we want to find  $\nabla_zf$
$$\dfrac{\partial{f}}{\partial{z}}=\dfrac{\partial{f}}{\partial{a}}\dfrac{\partial{a}}{\partial{z}}$$
and we know derivative of the ReLU is:
$$\dfrac{\partial{a}}{\partial{z}} = \begin{cases}1 & z_i > 0 \\ 0  & z_i \leq 0 \end{cases} = \begin{bmatrix}
I_{(z_1>0)} &0&0\\
0&I_{(z_2>0)}&0\\
0&0&I_{(z_3>0)}
\end{bmatrix}$$
so we can get 
$$\nabla_zf=\dfrac{\partial{f}}{\partial{z}}= \begin{bmatrix}2a_1 \cdot I(z_1 > 0) \\ 2a_2 \cdot I(z_2 > 0) \\ 2a_3 \cdot I(z_3 > 0) \end{bmatrix} = 
\begin{bmatrix}2a_1\ if\ z_1 > 0,\ else\  0 \\2a_2\ if\ z_2 > 0,\ else\  0 \\2a_3\ if\ z_3 > 0,\ else\  0  \end{bmatrix} = 2a\cdot I_{z>0}$$ Now to find $\nabla_x f$ 
$$\dfrac{\partial{f}}{\partial{x}}=\dfrac{\partial{f}}{\partial{z}}\dfrac{\partial{z}}{\partial{x}}$$
we know can find the $\dfrac{\partial{z}}{\partial{x}}$ which is :
$$\dfrac{\partial{z_k}}{\partial{x_i}}=W_{k,i}$$

so
$$\nabla_xf = 
\dfrac{\partial{f}}{\partial{x}}=
\sum_j\dfrac{\partial{f}}{\partial{z_j}}\dfrac{\partial{z_j}}{\partial{x_i}}=
\sum_j2z_i\cdot I_{(z_i>0)} W_{j,i} =
\nabla_zf \cdot W^T$$
on the other hand $\nabla_Wf$ is much easier to find
$$\nabla_Wf = \nabla_zf \cdot x^T$$  