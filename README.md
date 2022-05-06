
## Introduction
The properties of atoms, molecules, and materials are maybe the most fundamental aspect of understanding the world that we interact with.
Moreover, the design of new matter structures with desirable properties has been leading the technological advancement, with achievements in a wide spectrum of applications affecting our daily life, from lithium batteries used in our smartphones to drug discovery that can save lives. The applications of material design are astonishing.

In traditional science, new materials are discovered by either experimentation or theoretical calculations. Both processes are time-costly and highly inefficient. 
This is where Machine Learning enters. Enabling the capitalization of the data obtained by traditional methods sets new horizons in the material discovery field, which were inaccessible by traditional methods. 

## About the project

In this project, I have developed a Machine Learning model to predict the properties of small organic molecules. 
To train and validate my model, I have used the GDB7-13 Dataset, which can be found here: https://qmml.org/datasets.html. It contains 7k stable small organic molecules, consisting of up to 7 atoms of the elements C, N, O, S, and Cl, and It was generated for Machine Learning purposes (see here: https://doi.org/10.1088/1367-2630/15/9/095003).

The purpose of this project was two-fold. The first was to explore and improve past models for small organic molecules. Indeed, I reduced the RMSE up to 41%. 
The second goal was to be presented to a non-expert audience, for I developed a web application using Streamlit. 

### Local instalation of the web application

Prerequisites: Python3 and an updated web browser: Google Chrome, Firefox, Microsoft Edge, Safari.

1) To download the app, open a terminal and run the following command in the desired directory:
```sh
  git clone https://github.com/MariaDragoumi/ML_chem_organic.git 
```
2) Install Python packages:

The file **`requirments.txt`** contains a list of all the package versions and their dependencies, so it is recommended to create an environment and install the packages included in the list. To create an environment:
```sh
  python3 -m venv ML_chem_organic/env
```
To activate the environment:
```sh
  source ML_chem_organic/env/bin/activate
```
To install the necessary packages:
```sh
  pip install ML_chem_organic/requirments.txt
```

Another option is to install manually the packages below: 
- streamlit     1.7.0
- pandas        1.4.1
- sklearn       1.0.2
- matplotlib    3.5.1
- seaborn       0.11.2
- ase           3.22.1
- dscribe       1.2.1

3) To run the app change in the repo directory:
```sh
  cd ML_chem_organic
```
and type:
```sh
  streamlit run organic_molecules_app.py
```

This should open a web browser and the app should start running. This might take a while. After initialization is done the app is ready to use.


4) When finished, you can deactivate a virtual environment by typing:
```sh
deactivate
```



