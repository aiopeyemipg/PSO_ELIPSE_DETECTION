import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import svd

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("Ellipses/id_39.png")
if image is None:
    raise ValueError("Image not found. Check file name and path.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# ============================================================
# 2. EDGE DETECTION
# ============================================================

edges = cv2.Canny(gray, 100, 200)

plt.figure()
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis("off")
plt.show()

# Extract edge coordinates
edge_points = np.column_stack(np.where(edges > 0))
edge_points = np.flip(edge_points, axis=1)  # (row,col) → (x,y)

num_edges = len(edge_points)
print("Number of edge points detected:", num_edges)

if num_edges < 50:
    raise ValueError("Not enough edge points to select 50.")

# Scatter plot
plt.figure()
plt.title("Edge Points Scatter Plot")
plt.scatter(edge_points[:, 0], edge_points[:, 1], s=1)
plt.gca().invert_yaxis()
plt.show()

# ============================================================
# 3. FIT CONIC FROM N POINTS (GENERALIZED SVD METHOD)
# ============================================================

def fit_ellipse_from_points(points):
    M = np.column_stack([
        points[:, 0]**2,
        points[:, 0]*points[:, 1],
        points[:, 1]**2,
        points[:, 0],
        points[:, 1],
        np.ones(len(points))
    ])

    _, _, Vt = svd(M)
    params = Vt[-1]
    return params  # A,B,C,D,E,F


# ============================================================
# 4. CHECK IF CONIC IS ELLIPSE
# Condition: B² - 4AC < 0
# ============================================================

def is_ellipse(params):
    A, B, C, D, E, F = params
    return (B*B - 4*A*C) < 0


# ============================================================
# 5. FITNESS FUNCTION (Vectorized – Faster)
# ============================================================

def fitness(params):
    A, B, C, D, E, F = params

    x = edge_points[:, 0]
    y = edge_points[:, 1]

    vals = A*x*x + B*x*y + C*y*y + D*x + E*y + F
    total_error = np.sum(np.abs(vals))

    return 1 / (1 + total_error)


# ============================================================
# 6. PSO INITIALIZATION (50 POINTS PER PARTICLE)
# ============================================================

num_particles = 100
max_iter = 100
num_selected_points = 40

particles = []
pbest = []
pbest_scores = []
fitness_history = []

for _ in range(num_particles):
    idx = random.sample(range(num_edges), num_selected_points)
    particles.append(idx)

    pts = edge_points[idx]
    params = fit_ellipse_from_points(pts)

    if is_ellipse(params):
        score = fitness(params)
    else:
        score = 0

    pbest.append(idx)
    pbest_scores.append(score)

best_index = np.argmax(pbest_scores)
gbest = pbest[best_index][:]
gbest_score = pbest_scores[best_index]

print("Initial Best Fitness:", gbest_score)

# ============================================================
# 7. PSO MAIN LOOP (Discrete Mutation Version)
# ============================================================

for iteration in range(max_iter):

    for i in range(num_particles):

        # Copy current particle
        new_particle = particles[i][:]

        # Mutate one index
        replace_index = random.randint(0, num_selected_points - 1)
        new_particle[replace_index] = random.randint(0, num_edges - 1)

        pts = edge_points[new_particle]
        params = fit_ellipse_from_points(pts)

        if not is_ellipse(params):
            continue

        score = fitness(params)

        # Update personal best
        if score > pbest_scores[i]:
            pbest_scores[i] = score
            pbest[i] = new_particle[:]

        # Update global best
        if score > gbest_score:
            gbest_score = score
            gbest = new_particle[:]

        particles[i] = new_particle

    fitness_history.append(gbest_score)
    print(f"Iteration {iteration+1} | Best Fitness: {gbest_score}")

# ============================================================
# 8. FITNESS CONVERGENCE PLOT
# ============================================================

plt.figure()
plt.title("Fitness Convergence")
plt.plot(fitness_history)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()

# ============================================================
# 9. DRAW FINAL DETECTED ELLIPSE
# ============================================================

best_pts = edge_points[gbest]
ellipse = cv2.fitEllipse(best_pts.astype(np.int32))

result = image.copy()
cv2.ellipse(result, ellipse, (0, 255, 0), 2)

plt.figure()
plt.title("Final Detected Ellipse")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# ============================================================
# 10. SHOW BEST 50 SELECTED POINTS
# ============================================================

plt.figure()
plt.title("Best 50 Edge Points Used")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(best_pts[:, 0], best_pts[:, 1], s=40, c='yellow')
plt.axis("off")
plt.show()

print("\nFinal Best Fitness:", gbest_score)
