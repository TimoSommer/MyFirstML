# MyFirstML
A simple and general framework for running and analyzing machine learning methods on data. This repo is thought as the first good starting point for beginners in machine learning. It is focused on using sklearn models on tabular data. At the moment it supports only regression, but classification is easy to add. Many comments are included in the code to help understanding what is going on and how different machine learning models can be used.


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
 
### Rename the project
Rename the project to your liking. In PyCharm, right click on the project folder and select "Refactor" -> "Rename". This will rename the project folder and the .iml file. You can also rename the conda environment if you want.

## Get started:
The main machine learning script is located at MyFirstML/machine_learning/run_simple_ML.py. Try to execute it and see if everything runs smoothly. Then, use this script as starting point for all your machine learning.