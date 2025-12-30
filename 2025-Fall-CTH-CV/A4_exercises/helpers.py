from supplied import pflat, plot_camera, rital
import matplotlib.pyplot as plt
import numpy as np

def estimate_F_DLT(x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)

    Returns:
        F_tilde : 3x3 fundamental matrix estimated with DLT on normalized points
        s_min   : smallest singular value of M
        Mv_norm : || M v ||_2 for the solution vector v
    '''

    # Ensure homogeneous coordinates are normalized (last row = 1)
    x1 = pflat(x1s)
    x2 = pflat(x2s)

    assert x1.shape[1] == x2.shape[1], "x1s and x2s must have the same number of points"
    n = x1.shape[1]

    # Inhomogeneous coordinates
    x  = x1[0, :]
    y  = x1[1, :]
    xp = x2[0, :]
    yp = x2[1, :]

    # Build the M matrix (n x 9) for the eight-point algorithm
    # Each row corresponds to: [x' x, x' y, x', y' x, y' y, y', x, y, 1]
    M = np.zeros((n, 9))
    M[:, 0] = xp * x
    M[:, 1] = xp * y
    M[:, 2] = xp
    M[:, 3] = yp * x
    M[:, 4] = yp * y
    M[:, 5] = yp
    M[:, 6] = x
    M[:, 7] = y
    M[:, 8] = 1.0

    # Solve the homogeneous least-squares problem M v = 0 using SVD
    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1, :]          # Right singular vector corresponding to smallest singular value

    # Reshape to 3x3 matrix F_tilde
    F_tilde = v.reshape(3, 3)

    # Diagnostics: smallest singular value and ||M v||
    s_min = S[-1]
    Mv_norm = np.linalg.norm(M @ v)

    # print checks; comment out if not desired
    print("Minimum singular value of M:", s_min)
    print("Norm of Mv:", Mv_norm)

    return F_tilde, s_min, Mv_norm

def enforce_fundamental(F_approx):
    '''
    F_approx - Approximate Fundamental matrix (3x3)
    Returns:
        F_rank2 - Fundamental matrix with det(F_rank2) = 0 (rank-2 enforced)
    '''

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(F_approx)

    # Enforce rank-2 by setting the smallest singular value to zero
    S_enforced = np.diag(S)
    S_enforced[-1, -1] = 0.0

    # Reconstruct the fundamental matrix with rank 2
    F_rank2 = U @ S_enforced @ Vt

    return F_rank2

def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)

    Returns:
        E - Essential matrix with singular values (1, 1, 0)
    '''

    # SVD of the approximate essential matrix
    U, S, Vt = np.linalg.svd(E_approx)

    # Enforce the internal constraints:
    # two equal non-zero singular values (set to 1) and one zero
    S_new = np.diag([1.0, 1.0, 0.0])

    # Reconstruct the constrained essential matrix
    E = U @ S_new @ Vt

    return E

def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F   - Fundamental matrix (3x3)

    Returns:
        dists      - 1D array of point-to-line distances (pixels)
        mean_dist  - mean of these distances
    '''

    # Ensure homogeneous normalization
    x1 = pflat(x1s)
    x2 = pflat(x2s)

    # Epipolar lines in image 2 corresponding to points in image 1
    # l2_i = F x1_i
    l2 = F @ x1                # 3xN
    l2 = l2.T                  # N x 3, each row: [a, b, c]

    # Distances from x2 points to corresponding epipolar lines l2
    x2h = x2.T                 # N x 3
    num = np.abs(np.sum(l2 * x2h, axis=1))           # |l^T x|
    den = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2)         # sqrt(a^2 + b^2)
    dists = num / den                                  # N distances in pixels

    mean_dist = np.mean(dists)

    return dists, mean_dist

def convert_E_to_F(E, K1, K2):
    '''
    A function that gives you a fundamental matrix from an essential matrix and the two calibration matrices
    E  - Essential matrix (3x3)
    K1 - Calibration matrix for the first image (3x3)
    K2 - Calibration matrix for the second image (3x3)
    '''
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T   # (K2^{-1})^T = K2^{-T}

    F = K2_inv_T @ E @ K1_inv
    return F

def extract_P_from_E(E):
    '''
    A function that extracts the four P2 solutions
    E - Essential matrix (3x3)
    Returns:
        P - Array containing all four P2 solutions (4 x 3 x 4),
            where P[i, :, :] is the i-th 3x4 camera matrix
    '''

    # SVD of E: E = U * diag(1,1,0) * V^T  (approximately)
    U, S, Vt = np.linalg.svd(E)

    # Ensure det(U * V^T) > 0; if not, flip the sign of V
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    # Standard W matrix used in E decomposition
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    u3 = U[:, 2].reshape(3, 1)   # third column of U as a 3x1 vector

    P = np.zeros((4, 3, 4))

    # P2 = [U W V^T |  ±u3] and [U W^T V^T | ±u3]
    R1 = U @ W  @ Vt
    R2 = U @ W.T @ Vt

    P[0, :, :] = np.hstack((R1,  u3))
    P[1, :, :] = np.hstack((R1, -u3))
    P[2, :, :] = np.hstack((R2,  u3))
    P[3, :, :] = np.hstack((R2, -u3))

    return P

def triangulate_points_DLT(P1, P2, x1_tilde, x2_tilde):
    """
    Linear DLT triangulation for all point pairs for one camera pair.
    P1, P2 : 3x4 camera matrices
    x1_tilde, x2_tilde : 3xN normalized image points (homogeneous)
    Returns:
        X : 4xN homogeneous 3D points
    """
    n = x1_tilde.shape[1]
    X = np.zeros((4, n))

    for i in range(n):
        x1 = x1_tilde[:, i] / x1_tilde[2, i]
        x2 = x2_tilde[:, i] / x2_tilde[2, i]

        # Build A (4x4) for DLT: (u * p3^T - p1^T, v * p3^T - p2^T) for each camera
        A = np.zeros((4, 4))
        A[0] = x1[0] * P1[2] - P1[0]
        A[1] = x1[1] * P1[2] - P1[1]
        A[2] = x2[0] * P2[2] - P2[0]
        A[3] = x2[1] * P2[2] - P2[1]

        # Solve A X = 0 via SVD; take last right singular vector
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X[:, i] = X_h / X_h[3]    # homogeneous normalization

    return X