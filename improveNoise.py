import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import svd

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("id_247.png")
if image is None:
    raise ValueError("Image not found. Check file name and path.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ============================================================
# 2. IMPROVED PREPROCESSING (NEW)
# ============================================================

# Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5,5), 1.5)

# Adaptive Canny thresholds
median = np.median(blur)
lower = int(max(0, 0.66 * median))
upper = int(min(255, 1.33 * median))
edges = cv2.Canny(blur, lower, upper)

# Morphological closing (connect broken edges)
kernel = np.ones((3,3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

plt.figure()
plt.title("Improved Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis("off")
plt.show()

# ============================================================
# 3. DISTANCE TRANSFORM (NEW MAJOR IMPROVEMENT)
# ============================================================

dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

plt.figure()
plt.title("Distance Transform Map")
plt.imshow(dist_transform, cmap='jet')
plt.colorbar()
plt.axis("off")
plt.show()

# Extract edge coordinates
edge_points = np.column_stack(np.where(edges > 0))
edge_points = np.flip(edge_points, axis=1)

num_edges = len(edge_points)
print("Number of edge points detected:", num_edges)

# ============================================================
# 4. FIT CONIC FROM 5 POINTS
# ============================================================

def fit_ellipse_from_5pts(points):
    M = []
    for x, y in points:
        M.append([x*x, x*y, y*y, x, y, 1])
    M = np.array(M)

    U, S, Vt = svd(M)
    params = Vt[-1]
    return params

# ============================================================
# 5. CHECK IF CONIC IS ELLIPSE
# ============================================================

def is_ellipse(params):
    A, B, C, D, E, F = params
    return (B*B - 4*A*C) < 0

# ============================================================
# 6. ROBUST FITNESS FUNCTION (DISTANCE-BASED)
# ============================================================

def fitness(params):
    A, B, C, D, E, F = params

    total_distance = 0
    sample_points = 200

    for t in np.linspace(0, 2*np.pi, sample_points):
        # Parametric sampling using implicit gradient trick
        x = int(250 * np.cos(t) + 250)  # temporary circle guess
        y = int(250 * np.sin(t) + 250)

        if 0 <= x < dist_transform.shape[1] and 0 <= y < dist_transform.shape[0]:
            total_distance += dist_transform[y, x]
        else:
            total_distance += 50  # penalty if outside

    avg_distance = total_distance / sample_points

    return 1 / (1 + avg_distance)

# ============================================================
# 7. PSO INITIALIZATION
# ============================================================

num_particles = 100
max_iter = 100

particles = []
pbest = []
pbest_scores = []
fitness_history = []

for _ in range(num_particles):
    idx = random.sample(range(num_edges), 5)
    particles.append(idx)

    pts = edge_points[idx]
    params = fit_ellipse_from_5pts(pts)

    if is_ellipse(params):
        score = fitness(params)
    else:
        score = 0

    pbest.append(idx)
    pbest_scores.append(score)

gbest = pbest[np.argmax(pbest_scores)]
gbest_score = max(pbest_scores)

# ============================================================
# 8. PSO MAIN LOOP
# ============================================================

for iteration in range(max_iter):

    for i in range(num_particles):

        new_particle = particles[i][:]
        replace_index = random.randint(0, 4)
        new_particle[replace_index] = random.randint(0, num_edges - 1)

        pts = edge_points[new_particle]
        params = fit_ellipse_from_5pts(pts)

        if not is_ellipse(params):
            continue

        score = fitness(params)

        if score > pbest_scores[i]:
            pbest_scores[i] = score
            pbest[i] = new_particle[:]

        if score > gbest_score:
            gbest_score = score
            gbest = new_particle[:]

        particles[i] = new_particle

    fitness_history.append(gbest_score)
    print(f"Iteration {iteration+1} | Best Fitness: {gbest_score}")

# ============================================================
# 9. FITNESS CONVERGENCE
# ============================================================

plt.figure()
plt.title("Fitness Convergence")
plt.plot(fitness_history)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.show()

# ============================================================
# 10. DRAW FINAL ELLIPSE
# ============================================================

best_pts = edge_points[gbest]
ellipse = cv2.fitEllipse(best_pts.astype(np.int32))

result = image.copy()
cv2.ellipse(result, ellipse, (0, 255, 0), 2)

plt.figure()
plt.title("Final Detected Ellipse (Noise Robust)")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

print("Final Best Fitness:", gbest_score)
