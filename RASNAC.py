import cv2
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
RANSAC_ITERATIONS = 1000
DIST_THRESHOLD = 3
MIN_INLIERS = 30

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread("Ellipses/id_39.png", 0)

edges = cv2.Canny(image, 50, 150)
points = np.column_stack(np.where(edges > 0))
points = np.array([[p[1], p[0]] for p in points])  # convert (row,col) -> (x,y)

print("Total Edge Points:", len(points))


# =========================
# ELLIPSE DISTANCE FUNCTION
# =========================
def ellipse_distance(params, pts):
    xc, yc, a, b, theta = params
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = pts[:, 0] - xc
    y = pts[:, 1] - yc

    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t

    dist = (x_rot**2)/(a**2) + (y_rot**2)/(b**2)
    return np.abs(dist - 1)


# =========================
# FIT ELLIPSE FROM 5 POINTS
# =========================
def fit_ellipse_5pts(pts):
    try:
        ellipse = cv2.fitEllipse(pts.astype(np.float32))
        (xc, yc), (MA, ma), angle = ellipse

        a = MA / 2
        b = ma / 2
        theta = np.deg2rad(angle)

        return (xc, yc, a, b, theta)
    except:
        return None


# =========================
# MULTI ELLIPSE RANSAC
# =========================
detected_ellipses = []
remaining_points = points.copy()

while len(remaining_points) > MIN_INLIERS:

    best_ellipse = None
    best_inliers = []

    for _ in range(RANSAC_ITERATIONS):

        sample_idx = random.sample(range(len(remaining_points)), 5)
        sample_pts = remaining_points[sample_idx]

        ellipse = fit_ellipse_5pts(sample_pts)
        if ellipse is None:
            continue

        d = ellipse_distance(ellipse, remaining_points)
        inliers = remaining_points[d < DIST_THRESHOLD]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_ellipse = ellipse

    if best_ellipse is None or len(best_inliers) < MIN_INLIERS:
        break

    print("Detected ellipse with inliers:", len(best_inliers))
    detected_ellipses.append(best_ellipse)

    # Remove inliers
    mask = np.ones(len(remaining_points), dtype=bool)
    for pt in best_inliers:
        idx = np.where((remaining_points == pt).all(axis=1))[0]
        mask[idx] = False

    remaining_points = remaining_points[mask]


# =========================
# VISUALIZATION
# =========================
color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for ellipse in detected_ellipses:
    xc, yc, a, b, theta = ellipse
    cv2.ellipse(color_img,
                (int(xc), int(yc)),
                (int(a), int(b)),
                np.rad2deg(theta),
                0, 360,
                (0, 0, 255),
                2)

plt.imshow(color_img)
plt.title("Detected Ellipses")
plt.axis("off")
plt.show()
