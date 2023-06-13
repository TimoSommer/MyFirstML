"""
This script makes a linear toy dataset for testing purposes.
"""
from pathlib import Path
from sklearn.datasets import fetch_california_housing

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'regression', 'california_housing_toy_dataset.csv')

    data = fetch_california_housing(as_frame=True).frame

    data.to_csv(outpath, index=False)

