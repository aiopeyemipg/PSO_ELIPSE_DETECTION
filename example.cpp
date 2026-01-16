#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Generate random double in range [a, b]
double randomDouble(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(gen);
}

// Objective function: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
double objective_function(double x, double y) {
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}

// Particle structure
struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double best_cost;
};

// Initialize particles
std::vector<Particle> initialize_particles(
    int num_particles,
    const std::vector<std::pair<double, double>>& bounds
) {
    std::vector<Particle> particles;

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.best_cost = std::numeric_limits<double>::infinity();

        for (size_t d = 0; d < bounds.size(); ++d) {
            p.position.push_back(randomDouble(bounds[d].first, bounds[d].second));
            p.velocity.push_back(randomDouble(-1.0, 1.0));
        }

        p.best_position = p.position;
        particles.push_back(p);
    }

    return particles;
}

// Update velocity
void update_velocity(
    Particle& particle,
    const std::vector<double>& global_best_position,
    double w, double c1, double c2
) {
    for (size_t i = 0; i < particle.velocity.size(); ++i) {
        double r1 = randomDouble(0.0, 1.0);
        double r2 = randomDouble(0.0, 1.0);

        double cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i]);
        double social    = c2 * r2 * (global_best_position[i] - particle.position[i]);

        particle.velocity[i] = w * particle.velocity[i] + cognitive + social;
    }
}

// Update position
void update_position(
    Particle& particle,
    const std::vector<std::pair<double, double>>& bounds
) {
    for (size_t i = 0; i < particle.position.size(); ++i) {
        particle.position[i] += particle.velocity[i];

        // Enforce bounds
        particle.position[i] = std::max(bounds[i].first,
                               std::min(particle.position[i], bounds[i].second));
    }
}

// Particle Swarm Optimization
std::pair<std::vector<double>, double> particle_swarm_optimization(
    int num_particles,
    int max_iter,
    const std::vector<std::pair<double, double>>& bounds,
    double w, double c1, double c2
) {
    auto particles = initialize_particles(num_particles, bounds);

    std::vector<double> global_best_position;
    double global_best_cost = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter; ++iter) {
        for (auto& particle : particles) {
            double cost = objective_function(particle.position[0], particle.position[1]);

            // Update personal best
            if (cost < particle.best_cost) {
                particle.best_cost = cost;
                particle.best_position = particle.position;
            }

            // Update global best
            if (cost < global_best_cost) {
                global_best_cost = cost;
                global_best_position = particle.position;
            }
        }

        // Update velocity and position
        for (auto& particle : particles) {
            update_velocity(particle, global_best_position, w, c1, c2);
            update_position(particle, bounds);
        }

        std::cout << "Iteration " << iter + 1
                  << ": Best Cost = " << global_best_cost << std::endl;
    }

    return {global_best_position, global_best_cost};
}

int main() {
    // Algorithm parameters
    std::vector<std::pair<double, double>> bounds = {
        {-2.0, 2.0},   // x bounds
        {-1.0, 3.0}    // y bounds
    };

    int num_particles = 30;
    int max_iter = 100;
    double w = 0.5;
    double c1 = 1.5;
    double c2 = 1.5;

    auto result = particle_swarm_optimization(
        num_particles, max_iter, bounds, w, c1, c2
    );

    std::cout << "\nBest solution found:\n";
    std::cout << "x = " << result.first[0]
              << ", y = " << result.first[1]
              << ", Cost = " << result.second << std::endl;

    return 0;
}
