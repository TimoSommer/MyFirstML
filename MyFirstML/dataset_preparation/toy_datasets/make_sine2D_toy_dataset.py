"""
This script makes a 2D sine toy dataset for testing purposes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'sine2D_toy_dataset.csv')
    n_samples = 100

    x1 = np.linspace(0, 4*np.pi, n_samples)
    x2 = np.linspace(0, 4*np.pi, n_samples)
    y = np.sin(x1) + np.sin(x2)
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

    data.to_csv(outpath, index=False)

