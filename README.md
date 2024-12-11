# Solving computer vision problems with MINLP solvers

# What is this?

This repository shows how to use Mixed Integer Nonlinear Programming (MINLP) solvers to solve some common computer vision estimation problems to global optimality.

Many problems in computer vision are combinatorial and non-convex, making them difficult to solve to global optimality.


Modern open-source MINLP solvers such as SCIP and Cuenne are general enough to solve these kinds of problems with global optimality guarantees, unlike the commonly used local optimization methods such as Gauss-Newton.

We model and solve the problems in Python using `pyomo`, optionally exporting the problem instance to a file in AMPL and GAMS format.

The goal here is not to be real-time -- the branch-and-bound solvers are not tailored to the problems, and solution times from seconds to several minutes are normal.

But we can verify that a solution is correct and compare it to the solution from local methods.

## Problems 


### Maximum consensus: 

The maximum consensus problem for outlier-robust estimation, leads to difficult combinatorial optimization problems (1D): 

```math
\begin{equation}
	\begin{aligned}
		\argmax_{x \in \mathbb{R}, \, \mathbf{z}} \quad &\sum_{i=1}^N z_i\\
		\text{subject to} \quad  &z_i |x_i - x| \leq z_i \epsilon\\
		&z_i \in \{0, 1\}, \forall i \in \{1, ..., N\}\\
	\end{aligned}
\end{equation}
```

### Truncated Least Squares (TLS)

```math
\begin{equation}
	\begin{aligned}
	\argmin_{\mathbf{R} \in \mathrm{SO}(3) , \,\mathbf{t} \in \mathbb{R}^3} \quad \sum_{i=1}^{N} \min \left(||\mathbf{q}_i - \mathbf{R} \mathbf{p}_i + \mathbf{t}||_2^2, \epsilon^2 \right)
	\end{aligned}
\end{equation}
```

# Getting started

Easiest way to install the solvers is to use a conda environment:

```sh
conda create -n minlp-cv python==3.11 
conda activate minlp-cv
```

Then, install the recommended solver `SCIP`:

```sh
conda install conda-forge::scip

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

where `problem` can be `maximum_consensus_bilinear`, `tls_translation`, `tls_rotation`. `tls_rotation_dc`



