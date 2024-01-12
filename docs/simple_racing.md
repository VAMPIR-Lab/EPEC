# Race Craft

## How to pass?
## How to defend your position?
## What are passing guidelines?
### [Passing safely is the responsibility of the passer](https://youtu.be/RuvYtSfEG90?t=78)
### [Blocking is against the racing sporting code](https://youtu.be/60u3xPBYzQU?t=153)
- Formula E has the most overtakes
# f1

- $n_p$: number of players
<!--- $n^i_x$: number of states for player $i\in\{1,\cdots, n_p\}$-->
<!--- $n^i_u$: number of controls for player $i\in\{1,\cdots, n_p\}$-->
- $n_t$: number of time steps

Let $i$ refer to the $i^{\text{th}}$ player, and $k\in\{1,\dots,n_t\}$ refer to the $k^{\text{th}}$ time step. We define $x^i_k$ as the player state (position and velocity) for simple 2 DoF (degrees of freedom) motion in plane ($p\in\mathbb{R}^2$):

$$
x^i[k] = 
\begin{bmatrix}
p_1^i[k] \\
p_2^i[k] \\
\dot{p}_1^i[k] \\
\dot{p}_2^i[k] 
\end{bmatrix} \in \mathbb{R}^{4} \\
u^i[k] \in \mathbb{R}^2
$$
The player dynamics are defined in terms of time step size $\Delta_t$
$$
x^i[k+1] = \begin{bmatrix} 
p^i_1[k] + \Delta_t \;\dot{p}_1^i[k] + \frac{1}{2} \Delta_t^2 \; (u^i_1[k] - c_d \; \dot{p}_1^i[k])\\
p^i_2[k] + \Delta_t \;\dot{p}_2^i[k] + \frac{1}{2} \Delta_t^2 \; (u^i_2[k] - c_d \; \dot{p}_2^i[k])\\
\dot{p}_1^i[k] + \Delta_t \; (u^i_1[k] - c_d \; \dot{p}_1^i[k])\\
\dot{p}_2^i[k] + \Delta_t \; (u^i_2[k] - c_d \; \dot{p}_2^i[k])
\end{bmatrix}
$$

Define decision variable $Z \in \mathbb{R}^{(4+2) n_p n_t}$:
$$
Z = \begin{bmatrix}
x^1[1] \\
\cdots\\
x^1[n_t] \\
u^1[1] \\
\cdots\\
u^1[n_t] \\
\vdots\\
x^{n_p}[1] \\
\cdots\\
x^{n_p}[n_t] \\
u^{n_p}[1] \\
\cdots\\
u^{n_p}[n_t] 
\end{bmatrix}
$$


# Two-players ($a=1$ and $b=2$)

$$
\min f_a(Z) \\
\min f_b(Z)
$$

# $f_a(Z)$ or $f_1(Z)$
$$
f_a(Z) = \sum_{k=1}^{n_t} x_2^b[k]-2 x_2^a[k]  + \alpha^a_1\,  x_1^a[k]^2 +\alpha^a_2\, u^a[k]^T u^a[k]
$$

# $f_b(Z)$ or $f_2(Z)$
$$
f_b(Z) = \sum_{k=1}^{n_t} x_2^a[k]-2 x_2^b[k]  + \alpha^b_1\,  x_1^b[k]^2 +\alpha^b_2\, u^b[k]^T u^b[k]
$$