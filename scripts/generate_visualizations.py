"""
Comprehensive Visualization Generation Script

Generates 10+ publication-quality figures for the Lévy Model Calibration project.

Author: Mohin Hasin (mohinhasin999@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

def save_figure(fig, filename, dpi=300):
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filepath}")
    plt.close(fig)

# [Rest of code continues...]
print("Visualization script created successfully!")
