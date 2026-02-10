import cv2
import numpy as np
import random
from numpy.linalg import svd

# ===============================
# 1. Load image and detect edges
# ===============================

image = cv2.imread("ellips.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

edge_points = np.column_stack(np.where(edges > 0))
edge_points = np.flip(edge_points, axis=1)  # convert (row,col) → (x,y)

num_edges = len(edge_points)

# ===============================
# 2. Fit conic from 5 points
# ===============================

def fit_ellipse_from_5pts(points):
    M = []
    for x, y in points:
        M.append([x*x, x*y, y*y, x, y, 1])
    M = np.array(M)

    # Solve Mp = 0 using SVD
    U, S, Vt = svd(M)
    params = Vt[-1]

    return params  # [A,B,C,D,E,F]

# ===============================
# 3. Check if conic is ellipse
# ===============================

def is_ellipse(params):
    A, B, C, D, E, F = params
    return (B*B - 4*A*C) < 0

# ===============================
# 4. Fitness function
# ===============================

def fitness(params):
    A, B, C, D, E, F = params
    total_error = 0

    for x, y in edge_points:
        val = A*x*x + B*x*y + C*y*y + D*x + E*y + F
        total_error += abs(val)

    return 1 / (1 + total_error)

# ===============================
# 5. PSO Initialization
# ===============================

num_particles = 30
max_iter = 40

particles = []
pbest = []
pbest_scores = []

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

# ===============================
# 6. PSO Main Loop
# ===============================

for _ in range(max_iter):
    for i in range(num_particles):

        # Randomly modify one index
        new_particle = particles[i][:]
        replace_index = random.randint(0, 4)
        new_particle[replace_index] = random.randint(0, num_edges-1)

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

print("Best score:", gbest_score)

# ===============================
# 7. Draw Result
# ===============================

best_pts = edge_points[gbest]
best_params = fit_ellipse_from_5pts(best_pts)

# Convert conic to center/axes using OpenCV
ellipse = cv2.fitEllipse(best_pts.astype(np.int32))

cv2.ellipse(image, ellipse, (0,255,0), 2)
cv2.imshow("Detected Ellipse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
