import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import svd

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("ellipseTin.png")
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

# Scatter plot of edge points
plt.figure()
plt.title("Edge Points Scatter Plot")
plt.scatter(edge_points[:, 0], edge_points[:, 1], s=1)
plt.gca().invert_yaxis()
plt.show()

# ============================================================
# 3. FIT CONIC FROM 5 POINTS (SVD METHOD)
# ============================================================

def fit_ellipse_from_5pts(points):
    M = []
    for x, y in points:
        M.append([x*x, x*y, y*y, x, y, 1])
    M = np.array(M)

    U, S, Vt = svd(M)
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
# 5. FITNESS FUNCTION
# Uses algebraic distance from all edge points
# ============================================================

def fitness(params):
    A, B, C, D, E, F = params
    total_error = 0

    for x, y in edge_points:
        val = A*x*x + B*x*y + C*y*y + D*x + E*y + F
        total_error += abs(val)

    return 1 / (1 + total_error)


# ============================================================
# 6. PSO INITIALIZATION
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
# 7. PSO MAIN LOOP
# (Discrete mutation-based version)
# ============================================================

for iteration in range(max_iter):

    for i in range(num_particles):

        # Mutate one of the 5 indices
        new_particle = particles[i][:]
        replace_index = random.randint(0, 4)
        new_particle[replace_index] = random.randint(0, num_edges - 1)

        pts = edge_points[new_particle]
        params = fit_ellipse_from_5pts(pts)

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
# 10. SHOW BEST 5 SELECTED POINTS
# ============================================================

plt.figure()
plt.title("Best 5 Edge Points Used")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(best_pts[:, 0], best_pts[:, 1], s=80)
plt.axis("off")
plt.show()

print("Final Best Fitness:", gbest_score)
