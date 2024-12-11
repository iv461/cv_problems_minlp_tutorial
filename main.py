
"""
Here we model the consensus maximization (CM) and TLS problem as a mixed-integer 
linear or nonlinear problem (MINLP) using pyomo to evaluate off-the-shelf solvers.

We then can evaluate some globally optimal branch-and-bound solvers.
We can export the problem as GAMS and AMPL file to evaluate more solvers such as 
Couenne or BARON, by default we evaluate only SCIP.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from data_simulation import generate_registration_problem_from_pcl, create_rotation_estimation_problem
from rotation_geometry import compute_residual_angle_between_3d_rotations
import fire

from pyomo.environ import *

np.seterr(all="raise")
rng = np.random.default_rng(2873654)


def generate_data_for_pcl_registration_problem(num_points, noise_bound, outlier_rate, gt_translation):
    """
    Generate test data for a point cloud translation estimation problem. 
    The inliers are sampled from straight models, the outliers uniformly. 
    Additionally, a number of local minimas may be introduced
    """
    # Create a random line
    m = 7.65
    t = 17.53
    x = rng.uniform(0, 10, num_points)
    y = m * x + t
    inliers = np.stack([x, y]).T
    # print(f"inliers: {inliers.shape}")
    src, dst = generate_registration_problem_from_pcl(
        inliers, outlier_rate=outlier_rate, noise_bound=noise_bound)
    # Apply random transform
    dst += gt_translation
    return src, dst


def generate_3d_data(num_points, noise_bound, outlier_rate, gt_translation):
    inliers = np.vstack([rng.uniform(0, 10., num_points) for i in range(3)]).T
    src, dst = generate_registration_problem_from_pcl(
        inliers, outlier_rate=outlier_rate, noise_bound=noise_bound)
    # Apply random transform
    dst += gt_translation
    return src, dst


def generate_translation_votes(translation, outlier_rate, num_points, noise_bound, do_3d=False):
    if do_3d:
        src, dst = generate_3d_data(
            num_points, noise_bound, outlier_rate=outlier_rate, gt_translation=translation)
    else:
        src, dst = generate_data_for_pcl_registration_problem(num_points, noise_bound, outlier_rate=outlier_rate,
                                                              gt_translation=translation)
    # Create translational votes
    votes = dst - src
    intervals = np.concatenate(((votes - .5 * noise_bound)[..., np.newaxis],
                                (votes + .5 * noise_bound)[..., np.newaxis]), axis=-1)
    return intervals


def get_model_with_binary_constraints(centers, bound):
    """Here we formulate the consensus maximization as binary program (BLP)
    as in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    """
    N = centers.shape[0]
    model = ConcreteModel(name="ConsensusMaximizationBinary")
    model.x = Var(range(2))  # 1, 2, to enable broadcasting
    model.inliers = Var(range(N), within=Binary)

    model.constraints = ConstraintList()
    for i in range(N):
        model.constraints.add(
            model.inliers[i] * abs(centers[i, 0] - model.x[0]) <= model.inliers[i] * bound)
        model.constraints.add(
            model.inliers[i] * abs(centers[i, 1] - model.x[1]) <= model.inliers[i] * bound)
        # model.constraints.add(0 <= model.inliers[i])
        # model.constraints.add(model.inliers[i] <= 1)

    model.objective = Objective(
        expr=sum(model.inliers[i] for i in range(N)), sense=maximize)
    model.write("consensus_maximization_model_binary.nl",
                io_options={"symbolic_solver_labels": True})
    model.write("consensus_maximization_model_binary.gms",
                io_options={"symbolic_solver_labels": True})
    return model


def get_1d_model_with_binary_constraints(centers, bound):
    """Here we formulate the consensus maximization as binary  program (BLP)
    as in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    """
    N = centers.shape[0]
    model = ConcreteModel(name="ConsensusMaximizationBinary")
    model.x = Var(range(1))
    model.inliers = Var(range(N), within=Binary)

    model.constraints = ConstraintList()
    for i in range(N):
        model.constraints.add(
            model.inliers[i] * abs(centers[i, 0] - model.x[0]) <= model.inliers[i] * bound)

    model.objective = Objective(
        expr=sum(model.inliers[i] for i in range(N)), sense=maximize)
    model.write("consensus_maximization_1d_model_binary.nl",
                io_options={"symbolic_solver_labels": True})
    model.write("consensus_maximization_1d_model_binary.gms",
                io_options={"symbolic_solver_labels": True})
    return model


def get_model_with_bilinear_constraints(centers, bound):
    """Here we formulate the consensus maximization as a bilinear program (BLP)
    as in:
        "Consensus Set Maximization with Guaranteed Global 
                Optimality for Robust Geometry Estimation", Hongdong Li, ICCV 2009

    The inlier-variable is relaxed from binary to real in the interval of [0, 1]
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


def get_tls_smu(centers, bound):
    """

    """
    N = centers.shape[0]
    model = ConcreteModel(name="TLS-SMU")
    model.x = Var(range(1), bounds=[-10, 10], initialize=4.4535)
    # model.alpha = Var(range(1), bounds=[1e-8, 2], initialize=0.05)
    # model.inliers = Var(range(N), within=Binary)

    # SMU: Smooth approximation of max(x, 0) (i.e. ReLU) https://arxiv.org/pdf/2111.04682
    def smu_relu(x, a):
        # return (x + sqrt(x**2 + a**2)) / 2
        return (x + abs(x)) / 2

    # Difference of Convex (DC) functions reformulation of TLS:
    def tls_dc(r, a):
        return r - smu_relu(r - bound, a)

    # model.constraints = ConstraintList()
    # for i in range(N):
    #    model.constraints.add(model.inliers[i] * abs(centers[i, 0] - model.x[0]) <= model.inliers[i] * bound)
    def objective(model):
        return sum(tls_dc((centers[i, 0] - model.x[0])**2, 1e-4) for i in range(N))

    model.objective = Objective(rule=objective, sense=minimize)
    model.write("tls_1d_dc_smu.nl", io_options={
                "symbolic_solver_labels": True})
    model.write("tls_1d_dc_smu.gms", io_options={
                "symbolic_solver_labels": True})
    solver = SolverFactory('scip')
    solver.solve(model, tee=True)
    # Display the solution
    model.x.display()
    # print(f"x-val: {value(model.x)}")
    # result = np.array([value(model.x[i]) for i in model.N])
    return model


def read_from_gams_and_solve(model_filename):
    print(f"Loading model file {model_filename} ...")
    model = ConcreteModel()
    model_instance = model.create_instance(model_filename)

    solver = SolverFactory('couenne', executable=COUENNE_PATH)
    results = solver.solve(model_instance, tee=True)
    # Print results
    print("Optimal solution:")
    print("x =", [value(model.x[i]) for i in range(2)])
    print("Inliers:")
    # for i in range(centers.shape[0]):
    # print("Center", i+1, "is an inlier:", value(model.inliers[i]) > 1e-3)
    print("\nSolver termination condition:",
          results.solver.termination_condition)
    print("Solver status:", results.solver.status)
    print("Solver message:", results.solver.message)
    print("Solver statistics:")
    print("  Total time:", results.solver.time, "seconds")
    print("  Iterations:", results.solver.iterations)


def plot_problem(src, dst, votes):
    fig, axes = plt.subplots(2)

    # axes[0].scatter(inliers[:, 0], inliers[:, 1], c="red", alpha=.5, label="Inliers")
    axes[0].scatter(src[:, 0], src[:, 1], c="red", alpha=.5, label="Src")
    axes[0].scatter(dst[:, 0], dst[:, 1], c="green", alpha=.5, label="Dst")

    # SMU: Smooth approximation of max(x, 0) (i.e. ReLU) https://arxiv.org/pdf/2111.04682
    def smu_relu(x, a):
        return (x + abs(x)) / 2

    # Difference of Convex (DC) functions reformulation of TLS:
    def tls_dc(r, a):
        return r - smu_relu(r - .1, a)

    # model.constraints = ConstraintList()
    # for i in range(N):
    #    model.constraints.add(model.inliers[i] * abs(centers[i, 0] - model.x[0]) <= model.inliers[i] * bound)
    def objective(x):
        return sum(tls_dc((votes[i, 0] - x)**2, 1e-4) for i in range(len(votes[:, 0])))

    x = np.linspace(-10, 10, 300)
    y_vals = [objective(x_i) for x_i in x]

    axes[1].plot(x, y_vals, c="r", alpha=1., label="Votes")

    # axes[1].scatter(votes[:, 0], votes[:, 1], c="r", alpha=.005, label="Votes")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    # axes[0].scatter(dst[:, 0], dst[:, 1], c="grey", alpha=.5, label="Outliers")
    # plt.show()
    fig.suptitle(f"TLS linear regression")
    fig.legend()
    plt.savefig(f"tls_linear_regression.png", dpi=300)


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


def solve_with_pyomo(src, dst, gt_rot_mat, model):
    # solver = SolverFactory('couenne', executable=COUENNE_PATH)
    solver = SolverFactory('scip')
    results = solver.solve(model, tee=True)

    # Print results
    print("Optimal solution:")
    print("x =", [value(model.x[i]) for i in range(4)])

    solver = SolverFactory('scip')
    solver.solve(model, tee=True)
    # Display the solution
    model.x.display()

    rot_est = Rotation.from_quat(
        to_w_last([value(model.x[i]) for i in range(4)]))
    rot_mat_est = rot_est.as_matrix()

    residual_angle = math.degrees(
        compute_residual_angle_between_3d_rotations(gt_rot_mat, rot_mat_est))
    print(
        f"Residual angle between estimated rotation and ground-truth: {residual_angle:.3f}°")
    success = residual_angle < 0.1
    return success


def test_on_translation_problem():
    num_points = 500
    noise_bound = .1
    outlier_rate = .5

    trans = np.array([4.4535, 20.478])

    src, dst = generate_data_for_pcl_registration_problem(num_points, noise_bound, outlier_rate=outlier_rate,
                                                          gt_translation=trans)

    # Create translational votes (interval centers)
    votes = dst - src

    plot_problem(src, dst, votes)

    get_tls_smu(votes, noise_bound)
    return
    # best_x, inliers = solve_with_custom_bnb(votes, noise_bound)
    # num_inliers = np.count_nonzero(inliers)

    model_binary = get_model_with_binary_constraints(votes, noise_bound)
    model_bilinear = get_model_with_bilinear_constraints(votes, noise_bound)
    model_1d_binary = get_1d_model_with_binary_constraints(votes, noise_bound)
    model_1d_direct = create_cs_model_direct(votes, noise_bound)
    # read_from_gams_and_solve("consensus_maximization_model_binary.gms")
    # read_from_gams_and_solve("consensus_maximization_model_bilinear.gms")

def test_on_rotation_problem():
    noise_bound = .1
    data_scale = 1.
    num_inliers = 50
    outlier_rate = .6
    max_translation = 1.2
    adv_amount = .1
    src, dst, gt_rot_mat, adv_rotmat =\
        create_rotation_estimation_problem(
            num_inliers=num_inliers, noise_bound=(.5*noise_bound),
            outlier_rate=outlier_rate,
            second_largest_set_consensus_amount=adv_amount,
            dims=3, data_scale=data_scale, shuffle=True)

    model = create_quat_pyomo_model(src, dst, noise_bound)
    solve_with_pyomo(src, dst, gt_rot_mat, model)

if __name__ == "__main__":
    fire.Fire()
