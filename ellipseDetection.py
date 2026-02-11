import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ===============================
# 1. Load Image and Detect Edges
# ===============================

image = cv2.imread("ellips.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

height, width = edges.shape
edge_points = np.column_stack(np.where(edges > 0))

# ===============================
# 2. PSO PARAMETERS
# ===============================

num_particles = 30
max_iter = 50

w = 0.2
c1 = 1.5
c2 = 1.5

# Parameter bounds: [xc, yc, a, b, theta]
bounds = [
    (0, width),      # xc
    (0, height),     # yc
    (10, width//2),  # a
    (10, height//2), # b
    (0, math.pi)     # theta
]

# ===============================
# 3. Fitness Function
# ===============================

def ellipse_fitness(params):
    xc, yc, a, b, theta = params
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    count = 0
    samples = 200

    for t in np.linspace(0, 2*math.pi, samples):
        x = a * math.cos(t)
        y = b * math.sin(t)

        # Rotation
        xr = x * cos_t - y * sin_t + xc
        yr = x * sin_t + y * cos_t + yc

        xi = int(round(xr))
        yi = int(round(yr))

        if 0 <= xi < width and 0 <= yi < height:
            if edges[yi, xi] > 0:
                count += 1

    return count / samples  # maximize

# ===============================
# 4. Initialize Particles
# ===============================

particles = []
velocities = []
pbest = []
pbest_scores = []

for _ in range(num_particles):
    particle = [
        random.uniform(bounds[i][0], bounds[i][1])
        for i in range(5)
    ]
    velocity = [random.uniform(-1, 1) for _ in range(5)]
    
    particles.append(particle)
    velocities.append(velocity)
    pbest.append(particle[:])
    pbest_scores.append(ellipse_fitness(particle))

gbest = pbest[np.argmax(pbest_scores)]
gbest_score = max(pbest_scores)

# ===============================
# 5. PSO MAIN LOOP
# ===============================

for iteration in range(max_iter):
    for i in range(num_particles):
        score = ellipse_fitness(particles[i])
        
        # Update personal best
        if score > pbest_scores[i]:
            pbest_scores[i] = score
            pbest[i] = particles[i][:]
        
        # Update global best
        if score > gbest_score:
            gbest_score = score
            gbest = particles[i][:]
        
        # Update velocity and position
        for d in range(5):
            r1 = random.random()
            r2 = random.random()
            
            velocities[i][d] = (
                w * velocities[i][d]
                + c1 * r1 * (pbest[i][d] - particles[i][d])
                + c2 * r2 * (gbest[d] - particles[i][d])
            )
            
            particles[i][d] += velocities[i][d]
            
            # Clamp to bounds
            particles[i][d] = max(bounds[d][0],
                                   min(bounds[d][1], particles[i][d]))

print("Best parameters:", gbest)

# ===============================
# 6. Draw Final Ellipse
# ===============================

xc, yc, a, b, theta = gbest

cv2.ellipse(image,
            (int(xc), int(yc)),
            (int(a), int(b)),
            math.degrees(theta),
            0, 360,
            (0, 255, 0), 2)

plt.figure()
plt.title("Detected Ellipse")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
