# Solving computer vision problems using MINLP solvers

# What is this ? 

This repository shows how to use Mixed Integer Nonlinear Programing (MINLP) solvers to solve some common computer vision estimation problems to global optimality. 

Many problems in computer vision are combinatorial and non-convex, solving them global optimality is therefore difficult. 

Modern open-source MINLP-solvers such as SCIP and Cuenne are general enough so that they can solve these kind of problems, with global optmality guarantees, unlike the commonly used local optimization methods such as Gauss-Newton.

We model here the problems in Python using `pyomo` and solve them, optinaly exporting the problem instance to file in  AMPL and GAMS format is possible.

The aim here is not be real-time capable -- the Branch-and-Bound solvers are not tailored specificly

But we can verify wheter a solution is correct, and compare it to the solution from local methods. 

## Problems 


### Maximum consensus: 

The maximum consensus problem for outlier-robust registration, leads to difficult combinatorial optimization problems: 


### Maximum consensus: 

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



