import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_orientation(csv_path):
    # 1. Load Data
    df = pd.read_csv(csv_path, sep='\t')
    labels = df['labels'].values
    theta = np.deg2rad(df['theta'].values)
    radius = df['radius'].values

    # 2. Project to 2D
    # Standard Polar -> Cartesian
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x, y = y, x
    # 3. Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='red')
    
    # Draw the head circle
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    # Label points
    for i, label in enumerate(labels):
        plt.text(x[i], y[i], label, fontsize=12, ha='right')

    # Draw "Nose" (Standard Convention: Nose is usually +Y or +X)
    plt.arrow(0, 0, 0, 1.1, head_width=0.05, color='blue', alpha=0.3, label='Positive Y (Up)')
    plt.arrow(0, 0, 1.1, 0, head_width=0.05, color='green', alpha=0.3, label='Positive X (Right)')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.title("Electrode Orientation Check\nWhere is AF3/Fp1?")
    plt.grid(True)
    plt.legend()
    plt.show()

# Run it on your file
check_orientation("data/coords.csv") # or whatever your csv is named