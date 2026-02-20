import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. LOAD IMAGE
# ============================================================

imageID = 26

image_path = f"Ellipses/id_{imageID}.png"
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. EDGE / POINT EXTRACTION
# ============================================================

edges = cv2.Canny(gray, 100, 200)

points = np.column_stack(np.where(edges > 0))
points = np.flip(points, axis=1)  # (x, y)

N = len(points)
print("Edge points:", N)

# ============================================================
# 3. 5-POINT ELLIPSE FIT (ALGEBRAIC)
# ============================================================

def ellipse_from_5_points(pts):
    x = pts[:, 0]
    y = pts[:, 1]

    D = np.column_stack([
        x*x, x*y, y*y, x, y, np.ones(5)
    ])

    if np.linalg.matrix_rank(D) < 5:
        return None

    _, _, V = np.linalg.svd(D)
    A = V[-1]
    a, b, c, d, e, f = A

    if b*b - 4*a*c >= 0:
        return None

    xc = (2*c*d - b*e) / (b*b - 4*a*c)
    yc = (2*a*e - b*d) / (b*b - 4*a*c)

    theta = 0.5 * np.arctan2(b, a - c)

    up = 2*(a*xc*xc + b*xc*yc + c*yc*yc - f)
    down1 = (a + c) + np.sqrt((a - c)**2 + b*b)
    down2 = (a + c) - np.sqrt((a - c)**2 + b*b)

    if down1 <= 0 or down2 <= 0:
        return None

    A_len = np.sqrt(up / down1)
    B_len = np.sqrt(up / down2)

    if not np.isfinite(A_len) or not np.isfinite(B_len):
        return None

    return np.array([xc, yc, A_len, B_len, theta])

# ============================================================
# 4. BALANCED GEOMETRIC FITNESS FUNCTION
# ============================================================

def ellipse_fitness(p):
    xc, yc, a, b, theta = p

    H, W = gray.shape

    # Hard geometric validity only
    if a < 5 or b < 5:
        return 1e9

    if max(a, b) > 20:
        return 1e9

    if xc - a < 0 or xc + a > W:
        return 1e9
    if yc - b < 0 or yc + b > H:
        return 1e9

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = points[:, 0] - xc
    y = points[:, 1] - yc

    xr =  x*cos_t + y*sin_t
    yr = -x*sin_t + y*cos_t

    f = (xr*xr)/(a*a) + (yr*yr)/(b*b)
    dist = np.abs(f - 1)

    # Define inliers
    threshold = 0.1
    inliers = dist < threshold

    num_inliers = np.sum(inliers)

    # If no inliers, penalize softly (NOT hard reject)
    if num_inliers == 0:
        return 1e6

    mean_error = np.mean(dist[inliers])

    # PURE INLIER-BASED OBJECTIVE
    fitness = mean_error / (num_inliers + 1e-6)

    return fitness



# ============================================================
# 5. INITIALIZE PSO FROM 5-POINT ELLIPSES
# ============================================================

num_particles = 1000
max_iter = 100
dim = 5

H, W = gray.shape

particles = []
velocities = []

for _ in range(num_particles):

    # randomly select 20 edge points
    idx = np.random.choice(N, 20, replace=False)
    sample_pts = points[idx]

    # estimate initial parameters from sampled points
    xc = np.mean(sample_pts[:,0])
    yc = np.mean(sample_pts[:,1])

    a = np.std(sample_pts[:,0]) * 2
    b = np.std(sample_pts[:,1]) * 2

    theta = np.random.uniform(0, np.pi)

    particles.append([xc, yc, a, b, theta])
    velocities.append(np.zeros(dim))

particles = np.array(particles)
velocities = np.array(velocities)

pbest = particles.copy()
pbest_scores = np.array([ellipse_fitness(p) for p in particles])

g_idx = np.argmin(pbest_scores)
gbest = pbest[g_idx].copy()
gbest_score = pbest_scores[g_idx]

print("Initial best fitness:", gbest_score)

# ============================================================
# 6. PSO LOOP
# ============================================================

w, c1, c2 = 0.2, 2, 2
history = []

for it in range(max_iter):

    for i in range(num_particles):
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest[i] - particles[i])
            + c2 * r2 * (gbest - particles[i])
        )

        particles[i] += velocities[i]
        
        # Axis constraint (5 ≤ axis ≤ 20)
        particles[i, 2] = np.clip(particles[i, 2], 5, 20)
        particles[i, 3] = np.clip(particles[i, 3], 5, 20)

        # Angle normalization
        particles[i, 4] = np.mod(particles[i, 4], np.pi)



        score = ellipse_fitness(particles[i])

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest[i] = particles[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest = particles[i].copy()

    history.append(gbest_score)
    print(f"Iter {it+1:3d} | Best fitness = {gbest_score:.6f}")

# ============================================================
# 7. VISUALIZATION
# ============================================================

xc, yc, a, b, theta = gbest

t = np.linspace(0, 2*np.pi, 600)
x = a * np.cos(t)
y = b * np.sin(t)

xr = x*np.cos(theta) - y*np.sin(theta) + xc
yr = x*np.sin(theta) + y*np.cos(theta) + yc

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(points[:,0], points[:,1], s=3, c='red', alpha=0.4)
plt.plot(xr, yr, 'lime', linewidth=1)
plt.title("Balanced PSO Ellipse (Between Scattered Points)")
plt.axis("off")
plt.show()

# ============================================================
# 8. CONVERGENCE
# ============================================================

plt.figure()
plt.plot(history, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("PSO Convergence")
plt.grid(True)
plt.show()

# ============================================================
# 9. FINAL PARAMETERS
# ============================================================

print(f"\nFinal ellipse parameters for image {imageID}:")
print("Center:", (xc, yc))
print("Axes:", (a, b))
print("Angle (rad):", theta)
print("Final fitness:", gbest_score)

# ============================================================
# 10. FORMAT OUTPUT AS DICTIONARY
# ============================================================

# Ensure correct major/minor axis ordering
if a >= b:
    semi_major = a
    semi_minor = b
    angle = theta
else:
    semi_major = b
    semi_minor = a
    angle = theta + np.pi/2  # rotate orientation if swapped

ellipse_dict = {
    "center_x": float(xc),
    "center_y": float(yc),
    "semi_major_axis": float(semi_major),
    "semi_minor_axis": float(semi_minor),
    "orientation_angle_rad": float(angle % (2*np.pi))
}

print("\nFinal ellipse dictionary:")
print(ellipse_dict)

def get_ellipse_properties(json_path, image_id):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Search for the selected image_id
    for annotation in data["annotations"]:
        if annotation["image_id"] == image_id:
            
            # If there are multiple ellipses, take the first one
            ellipse = annotation["ellipses"][0]
            
            return {
                "center_x": ellipse["center_x"],
                "center_y": ellipse["center_y"],
                "semi_major_axis": ellipse["semi_major_axis"],
                "semi_minor_axis": ellipse["semi_minor_axis"],
                "orientation_angle_rad": ellipse["orientation_angle_rad"]
            }

    return "Image ID not found"


# Example usage
json_file = "annotations.json"
result = get_ellipse_properties(json_file, imageID)
print("Groundtruth ellipse Data")
print(result)


# ============================================================
# 11. NORMALIZATION & ACCURACY EVALUATION
# ============================================================

def normalize_angle_pi(theta):
    return theta % np.pi

# Normalize detected angle
detected = ellipse_dict.copy()
detected["orientation_angle_rad"] = normalize_angle_pi(detected["orientation_angle_rad"])

# Normalize ground truth angle
gt = result.copy()
gt["orientation_angle_rad"] = normalize_angle_pi(gt["orientation_angle_rad"])

# Scale ground truth spatial parameters
gt["center_x"] /= 2
gt["center_y"] /= 2
gt["semi_major_axis"] /= 2
gt["semi_minor_axis"] /= 2

# ============================================================
# 12. ACCURACY FUNCTION
# ============================================================

def compute_accuracy(det, gt, key):
    """Relative accuracy (%)"""
    if gt[key] == 0:
        return 0
    error = abs(det[key] - gt[key]) / abs(gt[key])
    return max(0, 1 - error) * 100  # clamp to [0,100]

# Compute accuracy for each parameter
acc_center_x = compute_accuracy(detected, gt, "center_x")
acc_center_y = compute_accuracy(detected, gt, "center_y")
acc_major = compute_accuracy(detected, gt, "semi_major_axis")
acc_minor = compute_accuracy(detected, gt, "semi_minor_axis")
acc_angle = compute_accuracy(detected, gt, "orientation_angle_rad")

# Average accuracy
avg_accuracy = np.mean([acc_center_x, acc_center_y, acc_major, acc_minor, acc_angle])

# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n==== Normalized Detected Ellipse ====")
print(detected)

print("\n==== Normalized Ground Truth Ellipse (scaled) ====")
print(gt)

print("\n==== Parameter Accuracy (%) ====")
print(f"Center X Accuracy: {acc_center_x:.2f}%")
print(f"Center Y Accuracy: {acc_center_y:.2f}%")
print(f"Semi-Major Axis Accuracy: {acc_major:.2f}%")
print(f"Semi-Minor Axis Accuracy: {acc_minor:.2f}%")
print(f"Orientation Angle Accuracy: {acc_angle:.2f}%")

print(f"\n==== Average Ellipse Accuracy: {avg_accuracy:.2f}% ====")


# ============================================
# VISUALIZE INLIERS OF BEST ELLIPSE
# ============================================

xc, yc, a, b, theta = gbest

cos_t = np.cos(theta)
sin_t = np.sin(theta)

x = points[:, 0] - xc
y = points[:, 1] - yc

xr =  x*cos_t + y*sin_t
yr = -x*sin_t + y*cos_t

f = (xr*xr)/(a*a) + (yr*yr)/(b*b)
dist = np.abs(f - 1)

inliers = dist < 0.1

inlier_points = points[inliers]
outlier_points = points[~inliers]

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Outliers in red
plt.scatter(outlier_points[:,0], outlier_points[:,1],
            s=5, c='red', alpha=0.3, label='Outliers')

# Inliers in blue
plt.scatter(inlier_points[:,0], inlier_points[:,1],
            s=10, c='blue', alpha=0.8, label='Inliers')

# Ellipse
t = np.linspace(0, 2*np.pi, 600)
x_ell = a * np.cos(t)
y_ell = b * np.sin(t)

xr_ell = x_ell*np.cos(theta) - y_ell*np.sin(theta) + xc
yr_ell = x_ell*np.sin(theta) + y_ell*np.cos(theta) + yc

plt.plot(xr_ell, yr_ell, 'lime', linewidth=2, label='Detected Ellipse')

plt.legend()
plt.title("Ellipse Fit with Inliers Visualization")
plt.axis("off")
plt.show()

print("Total edge points:", len(points))
print("Inlier count:", np.sum(inliers))
print("Inlier ratio:", np.sum(inliers)/len(points))
