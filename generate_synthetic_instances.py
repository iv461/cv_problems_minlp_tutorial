"""
Create synthetic data problem instances for point cloud registration.
"""
import numpy as np 
import math
import fire
from scipy.spatial.transform import Rotation
from dataclasses import dataclass 

np.seterr(all="raise")

# Very important: Here we fix the seed of the pseudo-random generator to make the data generation deterministic: The points of the point clouds
# are random with respect to each other, but in every execution of the program, we get the exact same point clouds.
rng = np.random.default_rng(2873654)

@dataclass
class RegistrationProblemParams:
    # Parameters for generating instances of synthetic point cloud registration problems.
    N: int # Total number of points, N = len(P) = len(Q)
    adversarial_suboptimality: float # The optimality of the local optimum based on the global optimum, from 0 to 1
    eps: float # Noise bound epsilon
    outlier_rate: float # Amount of outliers, 0 - 1
    data_scale: float

@dataclass
class RegistrationProblem:
    # A point cloud registration problem.
    params: RegistrationProblemParams # The params that were used for creation
    P: np.ndarray # P points, Nx3 array
    Q: np.ndarray # Q points, Nx3 arry
    R_gt: np.ndarray # Optimal rotation 
    t_gt: np.ndarray # Optimal translation
    R_adv: np.ndarray # Adversarial (local) optimum
    t_adv: np.ndarray # Adversarial (local) optimum
    inlier_mask: np.ndarray # The ground truth inliers mask (1 for inlier, 0 for outlier) at the ground-truth (R_gt, t_gt)
    inlier_ids: np.ndarray
    outlier_ids: np.ndarray

def sample_ball_uniformly(dims, radius, num_samples):
    # A simple method for uniform sampling a 3d-ball based on rejection-sampling. 
    # There is surely a provably correct method which is not based on rejection-sampling 
    # out there, but for 3D this suffices.
    num_samples = int(num_samples)
    points = None
    while points is None or len(points) < num_samples:
        # We sample twice the samples since the probability of a point being in the ball is apporx. 50%.
        new_points = np.hstack([rng.uniform(-radius, radius, 2 * num_samples) for i in range(dims)]).reshape(2 *num_samples, dims)
        # Reject points outside of the ball
        new_points = new_points[np.linalg.norm(new_points, axis=1) <= radius]
        if points is None:
            points = new_points
        else:
            points = np.concatenate([points, new_points])
        if len(points) == num_samples:
            break
        elif len(points) > num_samples: # We sampled more points
            points = points[:num_samples]

    return points

def sample_2d_rotations(num_samples):
    return [np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]]) for theta in rng.uniform(0, np.pi, num_samples)]

def sample_3d_rotations(num_samples):
    # Sampled uniformly from the set of 3D rotations and returns an array of rotation vectors.
    rotations = Rotation.random(num=num_samples, random_state=rng)
    return [rot.as_matrix() for rot in rotations]
    
def add_outliers_in_box(src, dst, outlier_rate, shuffle):
    assert src.shape == dst.shape
    assert len(src.shape) == 2
    assert outlier_rate < 1. and outlier_rate >= 0
    
    dims = src.shape[-1]
    if outlier_rate > 0.:
        num_outliers = int(round(len(src) / (1. / outlier_rate - 1.)))
        bb = np.amin(src, axis=0), np.amax(src, axis=0)
        outliers1 = np.vstack([rng.uniform(bb[0][i], bb[1][i], num_outliers) for i in range(dims)]).T
        outliers2 = np.vstack([rng.uniform(bb[0][i], bb[1][i], num_outliers) for i in range(dims)]).T
        src = np.concatenate([src, outliers1])
        dst = np.concatenate([dst, outliers2])
    # Permute to shuffle the outliers with inliers: Otherwise, the chain-graph approach 
    # can exploit this order: There is a clear distrinction between outlier tims and inlier tims. 
    # By shuffling however, we get the realisic scenario
    if shuffle:
        shuffled_indices = rng.permutation(np.arange(len(dst)))
        dst = dst[shuffled_indices]
        src = src[shuffled_indices]
    return src, dst

def add_measurement_noise_and_outliers(pcl, outlier_rate, noise_bound, shuffle=True):
    # Clones the pcl and add uniform measurement noise to every point. Then, it samples outlier corrs and shuffles them with the inliers.
    assert outlier_rate < 1. and outlier_rate >= 0
    assert noise_bound >= 0.
    
    dims = pcl.shape[-1]
    src = pcl.copy() 
    dst = pcl.copy() 

    if noise_bound > 0:
        measurement_noise1 = sample_ball_uniformly(dims, noise_bound, dst.shape[0])
        measurement_noise2 = sample_ball_uniformly(dims, noise_bound, dst.shape[0])
        #measurement_noise = np.vstack([rng.normal(0., noise_bound, dst.shape[0]) for i in range(dims)]).T
        src += measurement_noise1[:, :dims]
        dst += measurement_noise2[:, :dims]
    return add_outliers_in_box(src, dst, outlier_rate, shuffle)

def evaluate_tls_objective(src, dst, rotmat, translation_vec, noise_bound):
    # Computes the value of the truncated least squares objective function for given two paired point clouds
    # and a 3d pose.
    noise_bound_sq = noise_bound**2
    return np.sum((np.linalg.norm((rotmat @ src.T).T + translation_vec - dst, axis=-1)**2).clip(max=noise_bound_sq))


def create_point_cloud_registration_problem(params: RegistrationProblemParams, dims = 3, add_rotation = True, add_translation = True):
    """
    Generates test data for a rotation estimation problem. 

    num_inliers: The number of inliers for the ground-truth inlier set. Inliers are sampled from an uniform distribution.
    The estimated inlier set has *at least* as many inliers. If this is a large number, 
    it is likely that the true inlier sets contains some more inliers because the sampled outliers 
    can also randomly lie inside.

    noise_bound: The bound of the noise added, which is sampled randomly and uniformly from a ball.
            This is the ball's diameter.

    outlier_rate: Number between 0 and (excluded) 1 which is the amount of outliers based on the *total* number of points.
            Outliers are sampled from a unform distribution in the bounding-box of the inliers.

    second_largest_set_consensus_amount: Number between 0 and 1 indicatint how large the second-largest 
        consensus set is which is also sampled. This is used to verify that the algorithm is actually globally optimal. 
        This setting is also called "adversarial" because it deliberately creates another inlier set which is however rotated differently.
        Can be 1: In this case, it solely depends on the random noise which inlier is better (based on the truncated least-squares cost)
            as both inlier sets are equally big. 


    data_scale: The data is multiplied with this number. Controls the signal-to-noise ratio together with the noise-bound
        and is useful for testing whether the algorithm can deal with arbitrarily scaled data.

    Returns a tuple of the source point cloud, the target point cloud and the ground-truth rotation matrix 
        and the rotation matrix of the adversarial rotation. 

    TODO Ensure exact consensus set size by removing outliers which randomly vote for the ground-truth rotation.
    """
    num_inliers = int(math.ceil(params.N * (1. - params.outlier_rate)))
    
    shuffle = True 
    noise_bound = params.eps * .5
    assert dims == 2 or dims == 3, "Only 2D and 3D supported"
    assert params.N >= 3
    assert params.outlier_rate < 1. and params.outlier_rate >= 0
    assert params.data_scale > 0.
    
    inliers1 = np.vstack([rng.uniform(-1., 1., num_inliers) for i in range(dims)]).T    
    inliers1 *= params.data_scale
    
    src1, dst1 = add_measurement_noise_and_outliers(inliers1, outlier_rate=0,\
        noise_bound=noise_bound, shuffle=shuffle)

    if add_rotation:
        if dims == 2:
            random_rotmats = sample_2d_rotations(2)
        else:
            random_rotmats = [rotation.as_matrix() for rotation in Rotation.random(num=2, random_state=rng)]
    else:
        random_rotmats = [np.eye(dims, dims), np.eye(dims, dims)]
    if add_translation:
        t_gt = sample_ball_uniformly(dims, params.data_scale, 1)
    else:
        t_gt = np.zeros(dims)
    
    R_gt = random_rotmats[0]
    dst1 = (R_gt @ dst1.T).T + t_gt

    inliers2 = np.vstack([rng.uniform(-1., 1., int(num_inliers * params.adversarial_suboptimality)) for i in range(dims)]).T    
    inliers2 *= params.data_scale
    
    src2, dst2 = add_measurement_noise_and_outliers(inliers2, outlier_rate=0,\
        noise_bound=noise_bound, shuffle=shuffle)
    
    R_adv = random_rotmats[1]
    dst2 = (R_adv @ dst2.T).T + t_gt

    src = np.concatenate((src1, src2))
    dst = np.concatenate((dst1, dst2))
    src, dst = add_outliers_in_box(src, dst, params.outlier_rate, shuffle=shuffle)
    
    # Now check if by coninsicence the local optimum is acutally better than the global optimum.
    # This can happen precisely because the measurement noise is random
    tls_val_at_gt = evaluate_tls_objective(
        src, dst, R_gt, t_gt, noise_bound)
    tls_val_at_adv = evaluate_tls_objective(
        src, dst, R_adv, t_gt, noise_bound)

    if tls_val_at_adv < tls_val_at_gt: R_gt, R_adv = R_adv, R_gt

    inlier_mask = np.linalg.norm((R_gt @ src.T).T + t_gt - dst, axis=-1) <= params.eps
    inlier_ids = np.argwhere(inlier_mask)
    outlier_ids = np.argwhere(~inlier_mask).T.astype(float)
    return RegistrationProblem(params, src, dst, R_gt, t_gt, 
            R_adv, t_gt, inlier_mask, inlier_ids, outlier_ids)

if __name__ == "__main__":
    fire.Fire()
