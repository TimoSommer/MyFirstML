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
