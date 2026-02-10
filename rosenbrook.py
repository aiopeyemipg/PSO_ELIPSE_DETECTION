# AlgorithmAfternoon.com
import random

def objective_function(x, y):
    """ Objective function f(x, y) = (1 - x)^2 + 100(y - x^2)^2 """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def initialize_particles(num_particles, bounds):
    """ Initialize particles with random positions and velocities """
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

def update_velocity(particle, global_best_position, w, c1, c2):
    """ Update velocity based on cognitive and social components """
    for i in range(len(particle['velocity'])):
        cognitive = c1 * random.random() * (particle['best_position'][i] - particle['position'][i])
        social = c2 * random.random() * (global_best_position[i] - particle['position'][i])
        particle['velocity'][i] = w * particle['velocity'][i] + cognitive + social

def update_position(particle, bounds):
    """ Update position based on velocity and bounds """
    for i in range(len(particle['position'])):
        particle['position'][i] += particle['velocity'][i]
        # Ensure particle stays within bounds
        particle['position'][i] = max(bounds[i][0], min(particle['position'][i], bounds[i][1]))

def particle_swarm_optimization(objective_function, bounds, num_particles, max_iter, w, c1, c2):
    """ Perform Particle Swarm Optimization """
    # Initialize particles
    particles = initialize_particles(num_particles, bounds)
    global_best_position = None
    global_best_cost = float('inf')

    for iteration in range(max_iter):
        for particle in particles:
            # Evaluate the objective function
            cost = objective_function(particle['position'][0], particle['position'][1])
            # Update personal best
            if cost < particle['best_cost']:
                particle['best_position'] = particle['position'][:]
                particle['best_cost'] = cost
                # Update global best
                if cost < global_best_cost:
                    global_best_position = particle['position'][:]
                    global_best_cost = cost

        # Update velocity and position
        for particle in particles:
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle, bounds)

        # Report best cost found in this iteration
        print(f"Iteration {iteration + 1}: Best Cost = {global_best_cost}")

    return global_best_position, global_best_cost

# Algorithm parameters
bounds = [(-2, 2), (-1, 3)]  # bounds for x and y
num_particles = 30000
max_iter = 100
w = 0.5
c1 = 1.5
c2 = 1.5

# Run the Particle Swarm Optimization algorithm
best_position, best_cost = particle_swarm_optimization(objective_function, bounds, num_particles, max_iter, w, c1, c2)
print(f"Best solution: x = {best_position[0]}, y = {best_position[1]}, Cost = {best_cost}")