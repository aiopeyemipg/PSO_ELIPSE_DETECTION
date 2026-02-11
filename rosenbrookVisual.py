import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# 1. OBJECTIVE FUNCTION (Rosenbrock Function)
# ============================================================

def objective_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# ============================================================
# 2. INITIALIZATION
# ============================================================

def initialize_particles(num_particles, bounds):
    particles = []
    for _ in range(num_particles):
        particle = {
            'position': [random.uniform(bounds[dim][0], bounds[dim][1]) for dim in range(len(bounds))],
            'velocity': [random.uniform(-1, 1) for _ in range(len(bounds))],
            'best_position': None,
            'best_cost': float('inf')
        }
        particles.append(particle)
    return particles


# ============================================================
# 3. VELOCITY & POSITION UPDATE
# ============================================================

def update_velocity(particle, global_best_position, w, c1, c2):
    for i in range(len(particle['velocity'])):
        r1 = random.random()
        r2 = random.random()

        cognitive = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
        social = c2 * r2 * (global_best_position[i] - particle['position'][i])

        particle['velocity'][i] = w * particle['velocity'][i] + cognitive + social


def update_position(particle, bounds):
    for i in range(len(particle['position'])):
        particle['position'][i] += particle['velocity'][i]
        particle['position'][i] = max(bounds[i][0], min(particle['position'][i], bounds[i][1]))


# ============================================================
# 4. PSO WITH HISTORY STORAGE
# ============================================================

def particle_swarm_optimization(objective_function, bounds, num_particles, max_iter, w, c1, c2):

    particles = initialize_particles(num_particles, bounds)

    global_best_position = None
    global_best_cost = float('inf')

    fitness_history = []
    particle_positions_history = []

    for iteration in range(max_iter):

        current_positions = []

        for particle in particles:

            cost = objective_function(particle['position'][0],
                                      particle['position'][1])

            if cost < particle['best_cost']:
                particle['best_position'] = particle['position'][:]
                particle['best_cost'] = cost

                if cost < global_best_cost:
                    global_best_position = particle['position'][:]
                    global_best_cost = cost

            current_positions.append(particle['position'][:])

        particle_positions_history.append(current_positions)

        for particle in particles:
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle, bounds)

        fitness_history.append(global_best_cost)

        print(f"Iteration {iteration+1}: Best Cost = {global_best_cost}")

    return global_best_position, global_best_cost, fitness_history, particle_positions_history


# ============================================================
# 5. PARAMETERS
# ============================================================

bounds = [(-2, 2), (-1, 3)]
num_particles = 150   # reduced for visualization
max_iter = 100
w = 0.5
c1 = 1.5
c2 = 1.5


# ============================================================
# 6. RUN PSO
# ============================================================

best_position, best_cost, fitness_history, particle_positions_history = \
    particle_swarm_optimization(objective_function, bounds,
                                num_particles, max_iter, w, c1, c2)

print(f"\nBest solution: x = {best_position[0]}, y = {best_position[1]}")
print(f"Minimum Cost = {best_cost}")


# ============================================================
# 7. 3D SURFACE PLOT
# ============================================================

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)
ax.set_title("3D Surface of Rosenbrock Function")
plt.show()


# ============================================================
# 8. CONTOUR + FINAL BEST POSITION
# ============================================================

plt.figure()
plt.contour(X, Y, Z, levels=50)
plt.scatter(best_position[0], best_position[1],
            color='red', s=100, label="Global Best")
plt.title("Contour Plot with Final Best Position")
plt.legend()
plt.show()


# ============================================================
# 9. FITNESS CONVERGENCE
# ============================================================

plt.figure()
plt.plot(fitness_history)
plt.title("Fitness Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.show()


# ============================================================
# 10. PARTICLE DISTRIBUTION (FIRST vs LAST ITERATION)
# ============================================================

first_positions = np.array(particle_positions_history[0])
last_positions = np.array(particle_positions_history[-1])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.contour(X, Y, Z, levels=30)
plt.scatter(first_positions[:,0], first_positions[:,1], s=10)
plt.title("Particles at Iteration 1")

plt.subplot(1,2,2)
plt.contour(X, Y, Z, levels=30)
plt.scatter(last_positions[:,0], last_positions[:,1], s=10)
plt.scatter(best_position[0], best_position[1],
            color='red', s=100)
plt.title("Particles at Final Iteration")

plt.show()