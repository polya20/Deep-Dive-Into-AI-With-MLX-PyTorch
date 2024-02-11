import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the range of x values
x = np.linspace(0, 10, 100)

# Define the linear function (blue line)
linear = x

# Define the S-curve function (red line) with a steeper exponential growth
# Adjust the logistic function parameters for a steeper S-curve
k = 1.8  # This controls the steepness of the curve
x0 = 6   # The midpoint of the logistic function, slightly shifted to the right
L = 100   # The curve's maximum value, to match the previous scale

# Calculate the S-curve values
s_curve_steep = L / (1 + np.exp(-k * (x - x0)))

# Plot the curves
plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=linear, label='Linear Growth (Blue Line)')
sns.lineplot(x=x, y=s_curve_steep, label='Steep S-curve Growth (Red Line)', color='red')

plt.title('Comparison of Expected Career Growth vs. AI-Driven Reality')
plt.xlabel('Time')
plt.ylabel('Growth')
plt.legend()
plt.grid(True)

# Save the plot to a file
# output_file = 's_curve_growth_comparison.png'
# plt.savefig(output_file)

# Show the plot
plt.show()

