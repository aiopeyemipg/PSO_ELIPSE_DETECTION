import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("Ellipses/id_2.png")
if image is None:
    raise ValueError("Image not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img_h, img_w = gray.shape
print("Image size:", img_w, "x", img_h)

# ============================================================
# 2. EDGE / POINT EXTRACTION
# ============================================================

edges = cv2.Canny(gray, 100, 200)

points = np.column_stack(np.where(edges > 0))
points = np.flip(points, axis=1)  # (x, y)

N = len(points)
print("Edge points:", N)

# ============================================================
# 3. ELLIPSE VALIDITY CHECK (INSIDE IMAGE)
# ============================================================

def ellipse_inside_image(xc, yc, a, b, theta):

    t = np.linspace(0, 2*np.pi, 200)
    x = a * np.cos(t)
    y = b * np.sin(t)

    xr = x*np.cos(theta) - y*np.sin(theta) + xc
    yr = x*np.sin(theta) + y*np.cos(theta) + yc

    if (
        np.min(xr) < 0 or np.max(xr) > img_w or
        np.min(yr) < 0 or np.max(yr) > img_h
    ):
        return False

    return True


# ============================================================
# 4. N-POINT ELLIPSE FIT (Least Squares)
# ============================================================

def ellipse_from_n_points(pts):

    x = pts[:, 0]
    y = pts[:, 1]

    D = np.column_stack([
        x*x,
        x*y,
        y*y,
        x,
        y,
        np.ones(len(pts))
    ])

    if np.linalg.matrix_rank(D) < 5:
        return None

    _, _, V = np.linalg.svd(D)
    A = V[-1]
    a, b, c, d, e, f = A

    if b*b - 4*a*c >= 0:
        return None

    denom = (b*b - 4*a*c)
    if abs(denom) < 1e-12:
        return None

    xc = (2*c*d - b*e) / denom
    yc = (2*a*e - b*d) / denom

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

    # ---- HARD IMAGE BOUNDARY CHECK ----
    if not ellipse_inside_image(xc, yc, A_len, B_len, theta):
        return None

    return np.array([xc, yc, A_len, B_len, theta])


# ============================================================
# 5. PURE GEOMETRIC FITNESS
# ============================================================

def ellipse_fitness(p):

    xc, yc, a, b, theta = p

    if a < 3 or b < 3:
        return 1e9

    if not ellipse_inside_image(xc, yc, a, b, theta):
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

    if np.sum(inliers) < 20:
        return 1e8

    return np.mean(dist[inliers])


# ============================================================
# 6. INITIALIZE PSO FROM 50-POINT ELLIPSES
# ============================================================

num_particles = 60
max_iter = 120
dim = 5
sample_size = 30

particles = []
attempts = 0

while len(particles) < num_particles and attempts < 8000:

    idx = np.random.choice(N, sample_size, replace=False)
    ell = ellipse_from_n_points(points[idx])

    if ell is not None:
        particles.append(ell)

    attempts += 1

if len(particles) == 0:
    raise ValueError("No valid ellipse initialization found")

particles = np.array(particles)
velocities = np.zeros_like(particles)

pbest = particles.copy()
pbest_scores = np.array([ellipse_fitness(p) for p in particles])

g_idx = np.argmin(pbest_scores)
gbest = pbest[g_idx].copy()
gbest_score = pbest_scores[g_idx]

print("Initial best fitness:", gbest_score)

# ============================================================
# 7. PSO LOOP (WITH HARD BOUNDARY CHECK)
# ============================================================

w, c1, c2 = 0.6, 1.5, 2
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

        candidate = particles[i] + velocities[i]

        if ellipse_inside_image(
            candidate[0], candidate[1],
            candidate[2], candidate[3],
            candidate[4]
        ):
            particles[i] = candidate

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
plt.title("PSO Ellipse (Strict Image Boundary)")
plt.axis("off")
plt.show()

# ============================================================
# 9. FINAL PARAMETERS
# ============================================================

print("\nFinal ellipse parameters:")
print("Center:", (xc, yc))
print("Axes:", (a, b))
print("Angle (rad):", theta)
print("Final fitness:", gbest_score)
