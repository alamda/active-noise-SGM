import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Original code by Agnish
def calculate_correlation(spin_matrix):
    Lx, Ly = spin_matrix.shape
    max_distance = min(Lx, Ly) // 2
    correlation = np.zeros(max_distance)
    normalization = np.zeros(max_distance)

    # Calculate correlation function
    for x1 in range(Lx):
        for y1 in range(Ly):
            S0 = spin_matrix[x1, y1]
            for x2 in range(Lx):
                for y2 in range(Ly):
                    S_r = spin_matrix[x2, y2]
                    distance = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                    if distance < max_distance:
                        correlation[distance] += S0 * S_r
                        normalization[distance] += 1

    # Normalize the correlation function
    correlation /= normalization

    # Subtract the mean product of spins
    mean_spin = np.mean(spin_matrix)
    correlation -= mean_spin ** 2
    distances = np.arange(1,max_distance+1,1)
    return distances, correlation

# Example usage
z = np.load("samples_0_0.npy")
spin_matrix = np.squeeze(z)
distances, correlation = calculate_correlation(spin_matrix)

plt.plot(distances, correlation)
plt.show()
