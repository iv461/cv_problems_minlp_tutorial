"""
Here we model the consensus maximization (CM) and TLS problem as a mixed-integer 
linear or nonlinear problem (MINLP) using pyomo to evaluate off-the-shelf solvers.

We then can evaluate some globally optimal branch-and-bound solvers.
We can export the problem as GAMS and AMPL file to evaluate more solvers such as 
Couenne or BARON, by default we evaluate only SCIP.
"""

import numpy as np
from generate_synthetic_instances import create_point_cloud_registration_problem, RegistrationProblemParams
import fire
from functools import partial 
from pyomo.environ import *

def create_instance(rotation):
    adversarial_suboptimality = 0.
    N = 100
    eps = .5
    data_scale = 10.
    outlier_rate = .5

    problem_params = RegistrationProblemParams(N, adversarial_suboptimality, eps, outlier_rate, data_scale)
    problem_instance = create_point_cloud_registration_problem(problem_params, dims=3, add_rotation=rotation, add_translation=True)

    if not rotation:
        return (problem_instance.P - problem_instance.Q), eps
    return problem_instance.P, problem_instance.Q, eps

def create_max_consensus_problem(y, epsilon):
    """ Create the maximum consensus problem as binary linear program (BLP)
    as described in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    """
    print(f"y. shape: {y.shape}")
    N = y.shape[0]
    d = y.shape[1]
    model = ConcreteModel(name="ConsensusMaximizationBinary")
    model.x = Var(range(d))
    model.z = Var(range(N), within=Binary)

    model.constraints = ConstraintList()
    for i in range(N):
        for j in range(d):
            model.constraints.add(model.z[i] * abs(y[i, j] - model.x[j]) <= model.z[i] * epsilon)
    
    model.objective = Objective(expr=sum(model.z[i] for i in range(N)), sense=maximize)
    model.write(f"maximum_consensus_{d}d_model_binary.nl", io_options={"symbolic_solver_labels": True})
    model.write(f"maximum_consensus_{d}d_model_binary.gms", io_options={"symbolic_solver_labels": True})
    return model

def create_max_consensus_bilinear_problem(centers, bound):
    """Create the maximum consensus problem as a bilinear program, 
    as described in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    The binary variables are relaxed to be in the interval of [0, 1]
    It is equivalent to the binary program but may be easier to solve.
    """
    N = centers.shape[0]
    model = ConcreteModel(name="ConsensusMaximizationBinary")
    model.x = Var(range(2))  # 1, 2, to enable broadcasting
    model.inliers = Var(range(N))

    model.constraints = ConstraintList()
    for i in range(N):
        model.constraints.add(
            model.inliers[i] * abs(centers[i, 0] - model.x[0]) <= model.inliers[i] * bound)
        model.constraints.add(
            model.inliers[i] * abs(centers[i, 1] - model.x[1]) <= model.inliers[i] * bound)
        model.constraints.add(0 <= model.inliers[i])
        model.constraints.add(model.inliers[i] <= 1)

    model.objective = Objective(
        expr=sum(model.inliers[i] for i in range(N)), sense=maximize)
    model.write("consensus_maximization_model_bilinear.nl",
                io_options={"symbolic_solver_labels": True})
    model.write("consensus_maximization_model_bilinear.gms",
                io_options={"symbolic_solver_labels": True})
    return model


def create_cs_model_direct(centers, bound):
    """Here we formulate the consensus maximization as a bilinear program (BLP)
    as in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    The inlier-variable is relaxed from binary to real in the interval of [0, 1]
    It is equivalent to the binary program but may be easier to solve.
    """
    N = centers.shape[0]
    model = ConcreteModel(name="ConsensusMaximizationBinary")
    model.R = RangeSet(-100, 100)
    model.x = Var(model.R, range(1))  # 1, 2, to enable broadcasting
    model.inliers = Var(range(N))

    model.objective = Objective(expr=sum(
        (abs(model.x[0] - centers[i, 0]) <= bound) for i in range(N)), sense=maximize)
    model.write("consensus_maximization_model_direct.nl",
                io_options={"symbolic_solver_labels": True})
    model.write("consensus_maximization_model_direct.gms",
                io_options={"symbolic_solver_labels": True})
    return model


def create_tls_smu_dc_problemn(y, epsilon):
    """
    Creates the Truncated Least Squares (1D) translation estimation problem.
    modeled without binary variables.
    """
    N = y.shape[0]
    model = ConcreteModel(name="TLS-SMU")
    model.x = Var(range(1))

    # SMU: Smooth approximation of max(x, 0) (i.e. ReLU) https://arxiv.org/pdf/2111.04682
    def smu_relu(x): return (x + abs(x)) / 2
    # Difference of Convex (DC) functions reformulation of TLS:
    def tls_dc(r): return r - smu_relu(r - bound)

    def objective(model):
        return sum(tls_dc((centers[i, 0] - model.x[0])**2) for i in range(N))

    model.objective = Objective(rule=objective, sense=minimize)
    model.write("tls_1d_dc_smu.nl", io_options={"symbolic_solver_labels": True})
    model.write("tls_1d_dc_smu.gms", io_options={"symbolic_solver_labels": True})
    return model

# Now the rotation problems
def quat_mul(a, b):
    """
    Implements the formula for multiplying two quaternions. 
    We have to re-implement it for pyomo since we cannot use libraries there.
    We use the wxyz order.
    See e.g. https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h
    as a reference.
    """
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
            a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]]

def inv_quat(q):
    # Invert a quaternion
    return [q[0], -q[1], -q[2], -q[3]]

def point_to_quat(a):
    """
    Homogenizes a point to a quaternion
    """
    return [0, a[0], a[1], a[2]]

def quat_to_point(a): return [a[1], a[2], a[3]]

def rotate_point_by_quat(q, p):
    return quat_to_point(quat_mul(quat_mul(q, point_to_quat(p)), inv_quat(q)))

def to_w_last(q):
    return [q[1], q[2], q[3], q[0]]

def sq_norm(a): return a[0]**2 + a[1]**2 + a[2]**2
def sq_norm_q(a): return a[0]**2 + a[1]**2 + a[2]**2 + a[3]**2

def create_quat_pyomo_model(src, dst, noise_bound):
    """
    Creates a optimization problem for finding the rotation (parametrized as a quaternion)
    between two 3D point cloud using the Tuncated Least Squares (TLS) objective.
    Uses the TLS-cost from "TEASER" H. Yang et al., T-RO 2020.
    """
    assert src.shape == dst.shape
    N = src.shape[0]
    model = ConcreteModel(name="TLSRotationQuat")
    model.x = Var(range(4))  # the quat

    # SMU: Smooth approximation of max(x, 0) (i.e. ReLU) https://arxiv.org/pdf/2111.04682
    def smu_relu(x):
        return (x + abs(x)) / 2

    # Difference of Convex (DC) functions reformulation of TLS:
    def min_dc(r, b):
        return r - smu_relu(r - b)

    model.constraints = ConstraintList()
    # Unit-quaternion constraint
    model.constraints.add(sq_norm_q(model.x) == 1)

    def objective(model):
        return sum(
            min_dc(
                sq_norm(
                    dst[i] - rotate_point_by_quat(model.x, src[i])),
                noise_bound*noise_bound) for i in range(N))

    model.objective = Objective(rule=objective, sense=minimize)
    model.write("tls_rotation_quat.nl", io_options={
                "symbolic_solver_labels": True})
    model.write("tls_rotation_quat.gms", io_options={
                "symbolic_solver_labels": True})
    return model

def solve_with_pyomo(model):
    solver = SolverFactory('scip')
    solver.solve(model, tee=True)
    # Display the solution
    model.x.display()

def solve_problem(problem: str): 
    problem_factory = {"maximum_consensus": partial(create_max_consensus_problem, *create_instance(rotation=False)), 
                        "tls_rotation_dc" : partial(create_max_consensus_problem, *create_instance(rotation=False)) }

    if problem not in problem_factory.keys():
        print(f"Unknown problem: '{problem}', available ones: {list(problem_factory.keys())}")
        return 

    solve_with_pyomo(problem_factory[problem]())

if __name__ == "__main__":
    fire.Fire()
