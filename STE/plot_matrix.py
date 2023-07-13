import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the size of the matrix
MATRIX_SIZE = 10

# Generate random values for the matrix
matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE)

# Create the figure and the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the indices for each point in the matrix
x, y, z = np.indices((MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE))

# Flatten the matrix and the indices
matrix = matrix.flatten()
x = x.flatten()
y = y.flatten()
z = z.flatten()

# Plot the points
ax.scatter(x, y, z, c=matrix, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Matrix Plot')

# Show the plot
plt.show()
