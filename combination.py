import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("Ellipses/id_518.png")
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

if N < 5:
    raise ValueError("Not enough points to fit an ellipse")

# ============================================================
# 3. 5-POINT ELLIPSE FIT (ALGEBRAIC)
# ============================================================

def ellipse_from_5_points(pts):
    x = pts[:, 0]
    y = pts[:, 1]

    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones(5)])

    if np.linalg.matrix_rank(D) < 5:
        return None

    _, _, V = np.linalg.svd(D)
    a, b, c, d, e, f = V[-1]

    if b*b - 4*a*c >= 0:
        return None

    xc = (2*c*d - b*e) / (b*b - 4*a*c)
    yc = (2*a*e - b*d) / (b*b - 4*a*c)

    theta = 0.5 * np.arctan2(b, a - c)

    up = 2 * (a*xc*xc + b*xc*yc + c*yc*yc - f)
    down1 = (a + c) + np.sqrt((a - c)**2 + b*b)
    down2 = (a + c) - np.sqrt((a - c)**2 + b*b)

    if down1 <= 0 or down2 <= 0:
        return None

    A = np.sqrt(up / down1)
    B = np.sqrt(up / down2)

    if not np.isfinite(A) or not np.isfinite(B):
        return None

    return np.array([xc, yc, A, B, theta])

# ============================================================
# 4. FAST GEOMETRIC PRE-CHECK
# ============================================================

def valid_5_point_set(pts, min_area=10):
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )
    return area > min_area

# ============================================================
# 5. BALANCED FITNESS FUNCTION
# ============================================================

cx_mean, cy_mean = np.mean(points, axis=0)

def ellipse_fitness(p):
    xc, yc, a, b, theta = p

    if a < 5 or b < 5:
        return 1e9

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = points[:, 0] - xc
    y = points[:, 1] - yc

    xr =  x*cos_t + y*sin_t
    yr = -x*sin_t + y*cos_t

    f = (xr*xr)/(a*a) + (yr*yr)/(b*b)
    dist = np.abs(f - 1)

    inliers = dist < 0.15
    inlier_ratio = np.mean(inliers)

    if inlier_ratio < 0.35:
        return 1e8

    inside = np.sum(f < 1)
    outside = np.sum(f > 1)
    balance = abs(inside - outside) / len(f)

    center_penalty = np.hypot(xc - cx_mean, yc - cy_mean)

    return (
        np.mean(dist[inliers])
        + 0.7 * balance
        + 0.01 * center_penalty
    )

# ============================================================
# 6. EXHAUSTIVE 5-POINT COMBINATIONS (NO RANDOMNESS)
# ============================================================

MAX_PARTICLES = 50
MAX_KEEP = 200

candidates = []

print("Generating ellipse hypotheses...")

for pts5 in combinations(points, 5):
    pts5 = np.array(pts5)

    if not valid_5_point_set(pts5):
        continue

    ell = ellipse_from_5_points(pts5)
    if ell is None:
        continue

    score = ellipse_fitness(ell)
    candidates.append((score, ell))

    if len(candidates) > MAX_KEEP:
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:MAX_PARTICLES]

if len(candidates) == 0:
    raise RuntimeError("No valid ellipses found")

candidates.sort(key=lambda x: x[0])
particles = np.array([c[1] for c in candidates[:MAX_PARTICLES]])

print("Initial ellipse candidates:", len(particles))

# ============================================================
# 7. PARTICLE SWARM OPTIMIZATION (REFINEMENT)
# ============================================================

num_particles = len(particles)
dim = 5
max_iter = 100

velocities = np.zeros_like(particles)

pbest = particles.copy()
pbest_scores = np.array([ellipse_fitness(p) for p in particles])

g_idx = np.argmin(pbest_scores)
gbest = pbest[g_idx].copy()
gbest_score = pbest_scores[g_idx]

print("Initial best fitness:", gbest_score)

w, c1, c2 = 0.6, 0.5, 2
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

        particles[i, 2] = np.clip(particles[i, 2], 5, 3*np.std(points[:,0]))
        particles[i, 3] = np.clip(particles[i, 3], 5, 3*np.std(points[:,1]))
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
# 8. VISUALIZATION
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
plt.plot(xr, yr, 'lime', linewidth=2)
plt.title("Deterministic Exhaustive Ellipse Detection")
plt.axis("off")
plt.show()

# ============================================================
# 9. CONVERGENCE PLOT
# ============================================================

plt.figure()
plt.plot(history, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("PSO Convergence")
plt.grid(True)
plt.show()

# ============================================================
# 10. FINAL PARAMETERS
# ============================================================

print("\nFinal ellipse parameters:")
print("Center:", (xc, yc))
print("Axes:", (a, b))
print("Angle (rad):", theta)
print("Final fitness:", gbest_score)
