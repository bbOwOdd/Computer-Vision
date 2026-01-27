import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from project_helpers import get_dataset_info, correct_H_sign, homography_to_RT
import os
from pathlib import Path

# ----------------------------
# Plotting
# ----------------------------
def camera_center(R, t):
    # C = -R^T t
    return (-R.T @ t).reshape(3)

def plot_map_and_cameras(map_X, R_abs, t_abs, title="SfM reconstruction", save_path=None, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X = map_X
    ax.scatter(X[0], X[1], X[2], s=1)

    Cs = []
    for i in range(len(t_abs)):
        if t_abs[i] is None:
            continue
        Cs.append(camera_center(R_abs[i], t_abs[i]))
    if len(Cs) > 0:
        Cs = np.asarray(Cs)
        ax.scatter(Cs[:, 0], Cs[:, 1], Cs[:, 2], s=30)

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

def plot_zoom_percentile(map_X, R_abs, t_abs, dataset, save_dir="figures",
                         p_lo=5, p_hi=95, show=False):
    """
    map_X: (3,N)
    p_lo/p_hi: percentiles used to set zoom window (e.g., 5..95).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X = map_X
    lo = np.percentile(X, p_lo, axis=1)   # (3,)
    hi = np.percentile(X, p_hi, axis=1)   # (3,)

    # Optional: only plot points inside the window (cleaner)
    mask = np.all((X.T >= lo) & (X.T <= hi), axis=1)
    Xz = X[:, mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Xz[0], Xz[1], Xz[2], s=2)

    # plot camera centers too (may be outside limits; still helpful)
    Cs = []
    for i in range(len(t_abs)):
        if t_abs[i] is None:
            continue
        C = (-R_abs[i].T @ t_abs[i]).reshape(3)
        Cs.append(C)
    if len(Cs) > 0:
        Cs = np.asarray(Cs)
        ax.scatter(Cs[:, 0], Cs[:, 1], Cs[:, 2], s=40)

    # Set zoomed axis limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])  # 3D limit setter exists for Axes3D [web:23]

    ax.set_title(f"SfM dataset {dataset} zoom ({p_lo}-{p_hi} pct)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    out = save_dir / f"sfm_dataset_{dataset}_zoom_{p_lo}_{p_hi}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

# ----------------------------
# Basic linear algebra helpers
# ----------------------------
def to_h(x2):
    return np.vstack([x2, np.ones((1, x2.shape[1]))])


def normalize_points(K, x_pix_2xN):
    return np.linalg.inv(K) @ to_h(x_pix_2xN)  # (3,N)


def skew(v):
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -vz, vy],
                     [vz, 0, -vx],
                     [-vy, vx, 0]], dtype=float)


def triangulate(P1, P2, x1, x2):
    N = x1.shape[1]
    X = np.zeros((4, N), dtype=float)
    for i in range(N):
        x = x1[:, i]
        xp = x2[:, i]
        A = np.vstack([
            x[0] * P1[2] - P1[0],
            x[1] * P1[2] - P1[1],
            xp[0] * P2[2] - P2[0],
            xp[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        Xi = Vt[-1]
        X[:, i] = Xi / Xi[3]
    return X  # (4,N)


def cheirality_mask(P1, P2, X4):
    X1 = P1 @ X4
    X2 = P2 @ X4
    return (X1[2] > 0) & (X2[2] > 0)


# ----------------------------
# Feature extraction & matching
# ----------------------------
class FeatureFrontend:
    def __init__(self):
        if hasattr(cv2, "SIFT_create"):
            self.det = cv2.SIFT_create()
            self.norm = cv2.NORM_L2
            self.dtype = np.float32
        else:
            self.det = cv2.ORB_create(5000)
            self.norm = cv2.NORM_HAMMING
            self.dtype = np.uint8
        self.matcher = cv2.BFMatcher(self.norm)

    def detect(self, img_gray):
        kps, desc = self.det.detectAndCompute(img_gray, None)
        if desc is None:
            return [], None
        return kps, desc

    def match_ratio(self, descA, descB, ratio=0.75):
        if descA is None or descB is None:
            return []
        knn = self.matcher.knnMatch(descA, descB, k=2)
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good


# ----------------------------
# Essential / Homography helpers
# ----------------------------
def dlt_homography(x1, x2):
    N = x1.shape[1]
    A = []
    for i in range(N):
        X = x1[:, i]
        x, y, w = x2[:, i]
        A.append(np.hstack([np.zeros(3), -w * X, y * X]))
        A.append(np.hstack([w * X, np.zeros(3), -x * X]))
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)


def eight_point_E(x1, x2):
    N = x1.shape[1]
    A = np.zeros((N, 9), dtype=float)
    for i in range(N):
        X1 = x1[:, i]
        X2 = x2[:, i]
        A[i] = [
            X2[0] * X1[0], X2[0] * X1[1], X2[0] * X1[2],
            X2[1] * X1[0], X2[1] * X1[1], X2[1] * X1[2],
            X2[2] * X1[0], X2[2] * X1[1], X2[2] * X1[2],
        ]
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    return U @ np.diag([s, s, 0.0]) @ Vt


def sampson_dist(E, x1, x2):
    Ex1 = E @ x1
    Etx2 = E.T @ x2
    x2tEx1 = np.sum(x2 * Ex1, axis=0)
    denom = Ex1[0]**2 + Ex1[1]**2 + Etx2[0]**2 + Etx2[1]**2
    denom = np.maximum(denom, 1e-12)
    return (x2tEx1**2) / denom


def decompose_E(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=float)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)

    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1

    return [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]


def choose_RT_from_E(E, x1, x2):
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    best_count = -1
    for R, t in decompose_E(E):
        P2 = np.hstack([R, t])
        X4 = triangulate(P1, P2, x1, x2)
        c = int(np.sum(cheirality_mask(P1, P2, X4)))
        if c > best_count:
            best_count = c
            best = (R, t)
    return best


def ransac_E_or_H(x1, x2, epi_th, H_th, iters=2500, seed=1):
    rng = np.random.default_rng(seed)
    N = x1.shape[1]

    best_E, best_E_inl = None, None
    best_H, best_H_inl = None, None

    for _ in range(iters):
        if N >= 8:
            idx = rng.choice(N, size=8, replace=False)
            E = eight_point_E(x1[:, idx], x2[:, idx])
            d = sampson_dist(E, x1, x2)
            inl = d < (epi_th**2)
            if best_E_inl is None or np.sum(inl) > np.sum(best_E_inl):
                best_E, best_E_inl = E, inl

        if N >= 4:
            idx = rng.choice(N, size=4, replace=False)
            H = dlt_homography(x1[:, idx], x2[:, idx])
            H = correct_H_sign(H, x1, x2)
            x2p = H @ x1
            x2p /= x2p[2:3]
            err = np.linalg.norm((x2p[:2] - x2[:2]), axis=0)
            inl = err < H_th
            if best_H_inl is None or np.sum(inl) > np.sum(best_H_inl):
                best_H, best_H_inl = H, inl

    # Prefer E if reasonable, else fall back to H->(R,t)
    if best_E is not None and np.sum(best_E_inl) >= 30:
        R, t = choose_RT_from_E(best_E, x1[:, best_E_inl], x2[:, best_E_inl])
        return R, t, best_E_inl

    if best_H is not None:
        RTs = homography_to_RT(best_H)
        best = None
        best_count = -1
        inl = best_H_inl

        for k in range(2):
            Rk = RTs[k, :, :3]
            tk = RTs[k, :, 3:4]
            E = skew(tk.ravel()) @ Rk
            R2, t2 = choose_RT_from_E(E, x1[:, inl], x2[:, inl])

            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R2, t2])
            X4 = triangulate(P1, P2, x1[:, inl], x2[:, inl])
            c = int(np.sum(cheirality_mask(P1, P2, X4)))
            if c > best_count:
                best_count = c
                best = (R2, t2)

        if best is None:
            raise RuntimeError("Homography produced no valid pose.")
        return best[0], best[1], inl

    raise RuntimeError("RANSAC failed: no E/H found.")


# ----------------------------
# Translation from 2D-3D with known R (2-point RANSAC)
# ----------------------------
def estimate_t_from_2d3d_minimal(x3, X3, R):
    A = []
    b = []
    for i in range(x3.shape[1]):
        S = skew(x3[:, i])
        A.append(S)
        b.append(-S @ (R @ X3[:, i].reshape(3, 1)))
    A = np.vstack(A)
    b = np.vstack(b)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t.reshape(3, 1)


def reproj_err_norm(x3, X3, R, t):
     # Compare normalized image coords (x/z, y/z)
    Y = (R @ X3) + t
    y = Y[:2] / np.maximum(Y[2:3], 1e-12)
    x = x3[:2] / np.maximum(x3[2:3], 1e-12)
    return np.linalg.norm(y - x, axis=0)


def estimate_t_ransac(x3, X3, R, thresh, iters=3000, seed=0):
    # 2-point RANSAC
    rng = np.random.default_rng(seed)
    N = x3.shape[1]
    if N < 2:
        raise RuntimeError("Need at least 2 2D-3D correspondences.")

    best_inl = None
    best_t = None

    for _ in range(iters):
        idx = rng.choice(N, size=2, replace=False)
        t = estimate_t_from_2d3d_minimal(x3[:, idx], X3[:, idx], R)
        e = reproj_err_norm(x3, X3, R, t)
        inl = e < thresh
        if best_inl is None or np.sum(inl) > np.sum(best_inl):
            best_inl = inl
            best_t = t

    if best_inl is None or np.sum(best_inl) < 6:
        raise RuntimeError("Translation RANSAC failed: too few inliers.")

    # Refit with all inliers
    t_refit = estimate_t_from_2d3d_minimal(x3[:, best_inl], X3[:, best_inl], R)
    return t_refit, best_inl


# ----------------------------
# Main SfM pipeline
# ----------------------------
def run_sfm(dataset: int, plot: bool = True):
    K, img_names, init_pair, pixel_th = get_dataset_info(dataset)

    f = float(K[0, 0])
    epi_th = pixel_th / f   # thresholds in normalized units
    H_th = 3.0 * pixel_th / f
    trans_th = 3.0 * pixel_th / f

    imgs = []
    for p in img_names:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        imgs.append(im)

    frontend = FeatureFrontend()

    # Precompute features
    kps_all, desc_all = [], []
    for im in imgs:
        kps, desc = frontend.detect(im)
        kps_all.append(kps)
        desc_all.append(desc)

    # Relative pose between consecutive frames (for rotation chaining)
    R_rel = [None] * len(imgs)
    for i in range(1, len(imgs)):
        matches = frontend.match_ratio(desc_all[i - 1], desc_all[i], ratio=0.75)
        if len(matches) < 12:
            raise RuntimeError(f"Not enough matches between {i-1} and {i}.")

        pts1 = np.array([kps_all[i - 1][m.queryIdx].pt for m in matches], dtype=float).T
        pts2 = np.array([kps_all[i][m.trainIdx].pt for m in matches], dtype=float).T
        x1 = normalize_points(K, pts1)
        x2 = normalize_points(K, pts2)

        R, t, inl = ransac_E_or_H(x1, x2, epi_th=epi_th, H_th=H_th, iters=2500, seed=10 + i)
        R_rel[i] = R

    # Chain absolute rotations (R0 = I)
    R_abs = [np.eye(3)]
    for i in range(1, len(imgs)):
        R_abs.append(R_rel[i] @ R_abs[i - 1])

    # Initialize from init pair
    i1, i2 = init_pair
    matches_init = frontend.match_ratio(desc_all[i1], desc_all[i2], ratio=0.75)
    if len(matches_init) < 20:
        raise RuntimeError("Init pair: not enough matches.")

    pts1 = np.array([kps_all[i1][m.queryIdx].pt for m in matches_init], dtype=float).T
    pts2 = np.array([kps_all[i2][m.trainIdx].pt for m in matches_init], dtype=float).T
    x1 = normalize_points(K, pts1)
    x2 = normalize_points(K, pts2)

    R12, t12, inl12 = ransac_E_or_H(x1, x2, epi_th=epi_th, H_th=H_th, iters=4000, seed=123)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R12, t12])
    X4 = triangulate(P1, P2, x1[:, inl12], x2[:, inl12])
    good = cheirality_mask(P1, P2, X4)

    X_cam_i1 = (X4[:3, good] / X4[3:4, good])

    # Rotate points to world coords (translation gauge not fixed yet)
    X_world = (R_abs[i1].T @ X_cam_i1)

    # Save descriptors from i1 for reconstructed 3D points
    inlier_matches = [m for m, keep in zip(matches_init, inl12) if keep]
    inlier_matches = [m for m, keep in zip(inlier_matches, good) if keep]

    map_desc = np.array([desc_all[i1][m.queryIdx] for m in inlier_matches], dtype=frontend.dtype)
    map_X = X_world.copy()

    # Estimate translations (fix gauge: t(i1)=0)
    t_abs = [None] * len(imgs)
    t_abs[i1] = np.zeros((3, 1))

    def get_2d3d_corr(img_idx):
        matches = frontend.match_ratio(desc_all[img_idx], map_desc, ratio=0.75)
        if len(matches) < 12:
            return None
        pts2d = np.array([kps_all[img_idx][m.queryIdx].pt for m in matches], dtype=float).T
        X3 = np.array([map_X[:, m.trainIdx] for m in matches], dtype=float).T
        x3 = normalize_points(K, pts2d)
        return x3, X3

    for i in range(len(imgs)):
        if i == i1:
            continue

        corr = get_2d3d_corr(i)
        if corr is None:
            continue

        x3, X3 = corr
        t_i, inl = estimate_t_ransac(x3, X3, R_abs[i], thresh=trans_th, iters=3000, seed=100 + i)
        t_abs[i] = t_i

        # Grow map by triangulating between i-1 and i when both translations exist
        j = i - 1
        if j >= 0 and t_abs[j] is not None and t_abs[i] is not None:
            matches_ij = frontend.match_ratio(desc_all[j], desc_all[i], ratio=0.75)
            if len(matches_ij) >= 20:
                ptsj = np.array([kps_all[j][m.queryIdx].pt for m in matches_ij], dtype=float).T
                ptsi = np.array([kps_all[i][m.trainIdx].pt for m in matches_ij], dtype=float).T
                xj = normalize_points(K, ptsj)
                xi = normalize_points(K, ptsi)

                Pj = np.hstack([R_abs[j], t_abs[j]])
                Pi = np.hstack([R_abs[i], t_abs[i]])
                X4n = triangulate(Pj, Pi, xj, xi)
                ok = cheirality_mask(Pj, Pi, X4n)

                Xnew = X4n[:3, ok] / X4n[3:4, ok]
                if Xnew.shape[1] > 0:
                    keep_matches = [m for m, keep in zip(matches_ij, ok) if keep]
                    dnew = np.array([desc_all[j][m.queryIdx] for m in keep_matches], dtype=frontend.dtype)

                    map_X = np.hstack([map_X, Xnew])
                    map_desc = np.vstack([map_desc, dnew])

    n_cams = sum(t is not None for t in t_abs)
    print(f"Dataset {dataset}: map points={map_X.shape[1]}, localized cameras={n_cams}/{len(imgs)}")

    # --- Plot ---
    if plot:
        os.makedirs("figures", exist_ok=True)
        '''
        plot_map_and_cameras(
            map_X, R_abs, t_abs, 
            title=f"SfM dataset {dataset}",                             
            save_path=f"figures/sfm_dataset_{dataset}.png",
            show=False
        )
        '''
        plot_zoom_percentile(map_X, R_abs, t_abs, dataset=args.dataset, p_lo=5, p_hi=95, show=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run_sfm(args.dataset, plot=args.plot)
