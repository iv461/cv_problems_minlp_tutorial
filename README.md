# Solving computer vision problems with MINLP solvers

# What is this?

This repository shows how to use Mixed Integer Nonlinear Programming (MINLP) solvers to solve some common computer vision estimation problems to global optimality.

Many problems in computer vision are combinatorial and non-convex, making them difficult to solve to global optimality.


Modern open-source MINLP solvers such as SCIP and Cuenne are general enough to solve these kinds of problems with global optimality guarantees, unlike the commonly used local optimization methods such as Gauss-Newton.

We model and solve the problems in Python using `pyomo`, optionally exporting the problem instance to a file in AMPL and GAMS format.

The goal here is not to be real-time -- the branch-and-bound solvers are not tailored to the problems, and solution times from seconds to several minutes are normal.

But we can verify that a solution is correct and compare it to the solution from local methods.

## Problems 


### Maximum consensus (MC): 

The maximum consensus (MC) problem for outlier-robust estimation, leads to difficult combinatorial optimization problems (1D) [1]: 

```math

\begin{equation}
	\begin{aligned}
		\max_{x \in \mathbb{R}, \, \mathbf{z}} \quad &\sum_{i=1}^N z_i\\
		\text{subject to} \quad  &z_i |y_i - x| \leq z_i \epsilon\\
		&z_i \in \{0, 1\}, \forall i \in \{1, ..., N\}\\
	\end{aligned}
\end{equation}
```

with the measured data points $y_i$, the binary decision variables $z_i$ deciding whether the i-th measurement is an inlier, and measurement accuracy $\epsilon$.

### Truncated Least Squares (TLS)

Truncated Least Squares for outlier-robust rotation estimation between two point clouds [2]:

```math
\begin{equation}
	\begin{aligned}
	\min_{\mathbf{R} \in \mathrm{SO}(3)} \quad \sum_{i=1}^{N} \min \left(||\mathbf{q}_i - \mathbf{R} \mathbf{p}_i ||_2^2, \epsilon^2 \right)
	\end{aligned}
\end{equation}
```

### TLS without binary variables

The Truncated least squares combinatorial problem can modeled in two ways: with N additional binary variables similar to the MC problem, or without binary variables. 
With 
```math
\begin{equation}
	\begin{aligned}
		\min(r, \epsilon^2) = r- \max(r- \epsilon^2, 0)
	\end{aligned}
\end{equation}
```
and

```math
\begin{equation}
	\begin{aligned}
		\max(x, 0) = \frac{x + |x|}{2}
	\end{aligned}
\end{equation}
```

we obtain the TLS-DC-ABS problem without binary variables, that is equivalent to the TLS problem:
```math
\begin{equation}
	\begin{aligned}
		(\text{TLS-DC-ABS}) \quad \text{TLS}(x) = \sum_{i=1}^{N} r_i - \frac{r_i - \epsilon^2 + |r_i - \epsilon^2|}{2}
	\end{aligned}
\end{equation}
```
with $r_i = ||\mathbf{q}_i - \mathbf{R} \mathbf{p}_i ||_2^2$.

Without binary variables, the problem can be solved significantly faster, 0.7s instead of 20s with SCIP.


# Getting started

Easiest way to install the solvers is to use a conda environment:

```sh
conda create -n minlp-cv python==3.11 
conda activate minlp-cv
```

Then, install the recommended solver `SCIP`:

```sh
conda install gcg papilo scip soplex zimpl --channel conda-forge
```

and the other requirements: 


```sh
pip install -r requirements.txt
```

# Run problems

To run the examples on synthetic problem instances, run: 

```sh 
python main.py solve_problem <problem>
```

where `problem` can be `maximum_consensus`, `tls_translation`, `tls_rotation`


## License 

MIT

## References 

[1] *H. Li, “Consensus set maximization with guaranteed global optimality for
robust geometry estimation,” in 2009 IEEE 12th International Conference on
Computer Vision, 2009*

[2] *H. Yang and L. Carlone, “A quaternion-based certifiably optimal solution to
the Wahba problem with outliers,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019*