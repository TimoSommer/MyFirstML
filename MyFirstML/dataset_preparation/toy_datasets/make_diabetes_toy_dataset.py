"""
This script makes a linear toy dataset for testing purposes.
"""
from pathlib import Path
from sklearn.datasets import load_diabetes

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'regression', 'diabetes_toy_dataset.csv')

    data = load_diabetes(as_frame=True).frame

    data.to_csv(outpath, index=False)

