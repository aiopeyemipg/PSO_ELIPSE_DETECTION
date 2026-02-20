import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import random

# ==============================
# PARAMETERS
# ==============================

num_particles = 40
max_iter = 120
inlier_threshold = 0.05
min_cluster_points = 20

# ==============================
# FITNESS (Inlier Maximization)
# ==============================

def ellipse_inlier_fitness(params, points):

    xc, yc, a, b, theta = params

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = points[:, 0] - xc
    y = points[:, 1] - yc

    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t

    val = (x_rot / a) ** 2 + (y_rot / b) ** 2
    error = np.abs(val - 1)

    inliers = error < inlier_threshold

    return -np.sum(inliers), inliers


# ==============================
# PSO FOR ONE CLUSTER
# ==============================

def detect_ellipse_pso(points, img_w, img_h):

    dim = 5
    particles = np.zeros((num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    for i in range(num_particles):

        xc = random.uniform(0, img_w)
        yc = random.uniform(0, img_h)

        max_a = min(xc, img_w - xc)
        max_b = min(yc, img_h - yc)

        a = random.uniform(10, max(15, max_a))
        b = random.uniform(10, max(15, max_b))

        theta = random.uniform(0, np.pi)

        particles[i] = [xc, yc, a, b, theta]

    pbest = particles.copy()
    pbest_scores = []
    pbest_masks = []

    for p in particles:
        score, mask = ellipse_inlier_fitness(p, points)
        pbest_scores.append(score)
        pbest_masks.append(mask)

    pbest_scores = np.array(pbest_scores)

    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx]
    gbest_score = pbest_scores[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5

    for _ in range(max_iter):

        for i in range(num_particles):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocities[i]

            particles[i, 0] = np.clip(particles[i, 0], 0, img_w)
            particles[i, 1] = np.clip(particles[i, 1], 0, img_h)
            particles[i, 2] = np.clip(particles[i, 2], 5, img_w/2)
            particles[i, 3] = np.clip(particles[i, 3], 5, img_h/2)
            particles[i, 4] = np.mod(particles[i, 4], np.pi)

            score, mask = ellipse_inlier_fitness(particles[i], points)

            if score < pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score

        best_idx = np.argmin(pbest_scores)

        if pbest_scores[best_idx] < gbest_score:
            gbest = pbest[best_idx]
            gbest_score = pbest_scores[best_idx]

    _, final_mask = ellipse_inlier_fitness(gbest, points)

    return gbest, final_mask


# ==============================
# MAIN PIPELINE
# ==============================

image = cv2.imread("Ellipses/id_39.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

# ---- CONTOUR FILTERING ----
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

all_points = []

for cnt in contours:
    if len(cnt) > 30:
        pts = cnt.reshape(-1, 2)
        all_points.append(pts)

if len(all_points) == 0:
    print("No valid contours found.")
    exit()

all_points = np.vstack(all_points)

# ---- DBSCAN CLUSTERING ----
clustering = DBSCAN(eps=5, min_samples=10).fit(all_points)
labels = clustering.labels_

unique_labels = set(labels)
unique_labels.discard(-1)

img_h, img_w = gray.shape
detected = []

for label in unique_labels:

    cluster_points = all_points[labels == label]

    if len(cluster_points) < min_cluster_points:
        continue

    print(f"Processing cluster {label}, points:", len(cluster_points))

    params, mask = detect_ellipse_pso(cluster_points, img_w, img_h)

    if np.sum(mask) > min_cluster_points:
        detected.append(params)

# ==============================
# VISUALIZATION
# ==============================

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

for params in detected:

    xc, yc, a, b, theta = params

    t = np.linspace(0, 2*np.pi, 400)
    x = a * np.cos(t)
    y = b * np.sin(t)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = x * cos_t - y * sin_t + xc
    y_rot = x * sin_t + y * cos_t + yc

    plt.plot(x_rot, y_rot, 'r', linewidth=2)

plt.gca().invert_yaxis()
plt.title("Hybrid Contour + DBSCAN + PSO Ellipse Detection")
plt.show()
