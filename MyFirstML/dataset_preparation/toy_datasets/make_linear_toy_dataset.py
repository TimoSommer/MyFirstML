"""
This script makes a linear toy dataset for testing purposes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'linear_toy_dataset.csv')
    n_samples = 100

    x = np.linspace(0, 1, n_samples)
    y = x + 1
    data = pd.DataFrame({'x1': x, 'y': y})

    data.to_csv(outpath, index=False)

