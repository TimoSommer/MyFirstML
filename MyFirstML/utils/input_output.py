"""
Utilities for input and output
"""
import os

def make_new_output_directory(rootdir, label) -> str:
    """
    Get numbered output directory `outdir` in parent directory `outdirname`.
    """
    os.makedirs(rootdir, exist_ok=True)

    # Get number that's not already used for a directory.
    num = 0
    dir_list = os.listdir(rootdir)
    while 'results_{}'.format(num) in '\t'.join(dir_list):
        num += 1

    outdir = os.path.join(rootdir, f'results_{num}_{label}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return outdir