# MyFirstML
This repo is intended as a first starting point for beginners in machine learning. It provides essential scripts with the basic components in a machine learning pipeline. For simplicity, it uses sklearn models on tabular data, but can of course be generalized. Many comments are included in the code to help understanding what is going on and how different machine learning models can be used.

## Installation:
### Clone the repo:
```
git clone https://github.com/TimoSommer/MyFirstML
```

### Install the conda environment:
```
cd MyFirstML
conda env create -f MyFirstML.yaml
conda activate MyFirstML
```  

### Use as PyCharm project:
Open the project in PyCharm and set the interpreter to the conda environment you just created. As of now, this is important to be able to import the MyFirstML package. Therefore, simply running `python ...` would not work at the moment. The package must be imported as PyCharm project.

## How to use
The main machine learning script is located at MyFirstML/machine_learning/run_simple_ML.py. Try to execute it and see if everything runs smoothly. Then, use this script as starting point for all your machine learning.

## Current limitations
- In its current implementation, the script is made for regression problems, but classification problems are easy to add.