import matplotlib.pyplot as plt

# Define masses and confidence levels for different models
masses = [500, 750, 1000, 1250, 1500]

# Example confidence levels for three different models
confidence_levels_model1 = [0.95965, 0.9322, 0.62011, 0.5471, 0.51950]

# Plot the confidence levels for each model
plt.plot(masses, confidence_levels_model1, marker='o', label='Model 1')

# Label the axes
plt.xlabel('Mass [GeV]')
plt.ylabel('Confidence Level')
plt.title('Confidence Level vs Mass for Various Models')

# Show the legend
plt.legend()

# Include a grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Customize the plot to be in HEP style
plt.style.use('seaborn-whitegrid')
plt.tight_layout()

# Show the plot
plt.savefig("confidence.png")
plt.show()
