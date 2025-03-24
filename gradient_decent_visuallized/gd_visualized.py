import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic data: bedrooms vs price
np.random.seed(42)
bedrooms = np.random.randint(1, 5, 100)  # Number of bedrooms (1-4)
size = bedrooms * 400 + np.random.normal(0, 150, 100)  # Size in sqft (linear relationship with noise)
price = 50000 * bedrooms + np.random.normal(0, 30000, 100)  # Price with noise

# Normalize data
size = (size - size.mean()) / size.std()
price = (price - price.mean()) / price.std()

# Initialize parameters
m = 0.0  # Slope
b = 0.0  # Intercept
learning_rate = 0.01
iterations = 100
n = len(size)

# Store history for visualization
cost_history = []
m_history = []
b_history = []

# Gradient Descent implementation
for _ in range(iterations):
    # Predictions
    y_pred = m * size + b
    
    # Calculate cost (MSE)
    cost = (1/(2*n)) * np.sum((y_pred - price)**2)
    cost_history.append(cost)
    
    # Calculate gradients
    dm = (1/n) * np.sum((y_pred - price) * size)
    db = (1/n) * np.sum(y_pred - price)
    
    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db
    
    # Store parameter history
    m_history.append(m)
    b_history.append(b)

# Final predictions
final_pred = m * size + b

# Create visualization
plt.figure(figsize=(18, 5))

# 1. Data and Regression Line
plt.subplot(1, 3, 1)
plt.scatter(size, price, alpha=0.5)
plt.plot(size, final_pred, color='red', label=f'Final: y = {m:.2f}x + {b:.2f}')
plt.xlabel('Normalized Size')
plt.ylabel('Normalized Price')
plt.title('Size vs Price & Regression Line')
plt.legend()

# 2. Cost Function Surface
m_vals = np.linspace(m-1, m+1, 100)
b_vals = np.linspace(b-1, b+1, 100)
M, B = np.meshgrid(m_vals, b_vals)
costs = np.zeros(M.shape)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        costs[i,j] = (1/(2*n)) * np.sum((M[i,j] * size + B[i,j] - price)**2)

ax = plt.subplot(1, 3, 2, projection='3d')
ax.plot_surface(M, B, costs, cmap='viridis', alpha=0.7)
ax.scatter(m_history, b_history, cost_history, color='red', s=20)
ax.set_xlabel('Slope (m)')
ax.set_ylabel('Intercept (b)')
ax.set_zlabel('Cost')
ax.set_title('Cost Function Surface')

# 3. Contour Plot with Gradient Descent Path
plt.subplot(1, 3, 3)
plt.contour(M, B, costs, 20, cmap='viridis')
plt.colorbar()
plt.plot(m_history, b_history, 'rx-', markersize=5)
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.title('Contour Plot with Gradient Descent Path')

plt.tight_layout()
plt.show()

# Plot cost convergence
plt.figure(figsize=(8, 5))
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()