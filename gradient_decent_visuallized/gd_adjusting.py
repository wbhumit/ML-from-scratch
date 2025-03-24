import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode

# Load data from CSV
try:
    df = pd.read_csv("C:/Users/wbhum/Documents/Projects/DS_ML_AI/Assignments/gradient_decent_visuallized/ols_from_scratch_data.csv")
    size = df['size'].values
    price = df['price'].values
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Normalize data
def normalize(x):
    return (x - x.mean()) / x.std()

size_norm = normalize(size)
price_norm = normalize(price)

# Initialize parameters
m = np.random.randn()  # Random initial slope
b = np.random.randn()  # Random initial intercept
learning_rate = 0.01
iterations = 50
n = len(size)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Gradient Descent Visualization with Real Data', fontsize=14)

# Store parameter history
m_history = [m]
b_history = [b]

for iteration in range(iterations):
    # Calculate predictions
    y_pred = m * size_norm + b
    
    # Calculate gradients
    dm = (-2/n) * np.sum(size_norm * (price_norm - y_pred))
    db = (-2/n) * np.sum(price_norm - y_pred)
    
    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db
    
    # Store parameter history
    m_history.append(m)
    b_history.append(b)
    
    # Update plots
    ax1.cla()
    ax2.cla()
    
    # Data and regression line plot
    ax1.scatter(size_norm, price_norm, alpha=0.5, label='Actual prices')
    ax1.plot(size_norm, y_pred, color='red', lw=2, 
            label=f'Current fit: y = {m:.2f}x + {b:.2f}')
    ax1.set_title(f'Iteration {iteration+1}/{iterations}')
    ax1.set_xlabel('Normalized Size')
    ax1.set_ylabel('Normalized Price')
    ax1.legend()
    
    # Parameter space plot
    ax2.plot(m_history, b_history, 'bo-', markersize=4)
    ax2.set_title('Parameter Space Trajectory')
    ax2.set_xlabel('Slope (m)')
    ax2.set_ylabel('Intercept (b)')
    ax2.grid(True)
    
    # Add gradient vector annotation
    ax2.annotate('', xytext=(m_history[-2], b_history[-2]),
                xy=(m_history[-1], b_history[-1]),
                arrowprops=dict(arrowstyle='->', color='r', lw=1.5))
    
    # Add cost annotation
    current_cost = np.mean((y_pred - price_norm)**2)
    ax1.annotate(f'Cost: {current_cost:.4f}\nLearning Rate: {learning_rate}',
                xy=(0.05, 0.85), xycoords='axes fraction',
                bbox=dict(boxstyle='round', alpha=0.8))
    
    plt.pause(0.1)  # Pause to see updates

# Finalize plots
plt.ioff()
plt.tight_layout()
plt.show()

# Show final results in original scale
# Calculate denormalized parameters
size_mean = size.mean()
size_std = size.std()
price_mean = price.mean()
price_std = price.std()

final_m = m * (price_std/size_std)
final_b = (price_mean - final_m * size_mean)

print(f"\nFinal equation in original scale:")
print(f"Price = {final_m:.2f} * Size + {final_b:.2f}")

# Plot final result in original units
plt.figure(figsize=(8, 6))
plt.scatter(size, price, alpha=0.5)
plt.plot(size, final_m * size + final_b, color='red', 
        label=f'Final: Price = {final_m:.2f}*Size + {final_b:.2f}')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Final Regression Fit (Original Scale)')
plt.legend()
plt.show()