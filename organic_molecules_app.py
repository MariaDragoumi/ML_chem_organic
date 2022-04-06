# import std libraries
from functools import cache
from math import degrees, gamma
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import random
from dscribe.descriptors import SOAP
from ase import io
from ase.db import connect
from ase.visualize import view
from ase import Atoms
from ase.calculators.aims import Aims
import os
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle


st.write('''
# Machine Learning in Chemistry:
## Prediction of organic molecule properties

---
''')

st.image('/Users/maria/Documents/spiced/spiced_projects/ML_chem/ML_chemistry/star_molecules.webp')
st.write('[Image Source](https://www.nature.com/articles/s41570-020-0189-9)')


expander_bar_what = st.expander("What?")
expander_bar_what.image('Presi Artworks_01/Maria Final Spiced Presi_01-01.png')

expander_bar_why = st.expander("Why?")
expander_bar_why.image('Presi Artworks_01/Maria Final Spiced Presi_01-02.png')
expander_bar_why.image('Presi Artworks_01/Maria Final Spiced Presi_01-03.png')

# expander_bar_why.markdown("""
# this is why

#     <h3 align="center">Current stage: Molecule configuration &#8594; Property</h3>

#     <h3 align="center">Goal: Property &#8594; Molecule configuration</h3> 
#     <h3 align="center">	&#8615;</h3>
#     <h3 align="center">Compound design</h3>
# """)
# #st.components.v1.html(raw_html,height=200)
expander_bar_how = st.expander("How?")
expander_bar_how.image('Presi Artworks_01/Maria Final Spiced Presi_01-04.png')
# expander_bar_why.markdown("""
# Machine Learning based on high accuracy (super expensive) Quantum Chemistry calculations
# """)

# raw_html = """
#     <p align="center">(Features: Molecule configurations, Labels: QC acurate properties) &#8594; ML model</p>
# """
# st.components.v1.html(raw_html,height=100)
# db_solar = connect('solar.db')
# random_molecule = db_solar.get_atoms('id=200')
# random_molecule.translate(-random_molecule.get_center_of_mass())
# html_item = view(random_molecule, viewer='x3d')
# raw_html = html_item._repr_html_()
# st.components.v1.html(raw_html, width=700,height=400)

@st.cache
def get_geometries():
    with open('gdb7-13/dsgdb7njp.xyz') as f:
        geometries = f.read()
    geometries = geometries.split('\n\n')
    return geometries

@st.cache(allow_output_mutation=True)
def molecules_lists(geometries):
    val_set_index = np.random.choice(range(len(geometries[1:])),size=int(len(geometries)/3))
    small_molecules = []
    targets = []
    val_small_molecules = []
    val_targets = []
    for i, geometry in enumerate(geometries):
        if i not in val_set_index:
            with open(f'geometries/{i}.xyz','w') as f:
                f.write(geometry)
            small_molecules.append(io.read(f'geometries/{i}.xyz'))
            targets.append([float(val) for val in geometry.split('\n')[1].split(' ')])
            os.remove(f'geometries/{i}.xyz')
        else:
            with open(f'geometries/{i}.xyz','w') as f:
                f.write(geometry)
            val_small_molecules.append(io.read(f'geometries/{i}.xyz'))
            val_targets.append([float(val) for val in geometry.split('\n')[1].split(' ')])
            os.remove(f'geometries/{i}.xyz')  
    return small_molecules, targets, val_small_molecules, val_targets         

def plot_molecule(molecule):
    molecule.translate(-molecule.get_center_of_mass())
    html_item = view(molecule, viewer='x3d')
    raw_html = html_item._repr_html_()
    st.components.v1.html(raw_html, height=400)

@st.cache(allow_output_mutation=True)
def get_targets(small_molecules, targets, val_small_molecules, val_targets):
    columns=['ae_pbe0', 'p_pbe0', 'p_scs', 'homo_gw', 'homo_pbe0', 'homo_zindo', 'lumo_gw', 'lumo_pbe0', 'lumo_zindo', 'ip_zindo', 'ea_zindo', 'e1_zindo', 'emax_zindo', 'imax_zindo']
    chemical_formulas = []
    for mol in small_molecules:
        chemical_formulas.append(mol.get_chemical_formula())
    val_chemical_formulas = []
    for mol in val_small_molecules:
        val_chemical_formulas.append(mol.get_chemical_formula())
    targets_df =pd.DataFrame(targets, columns=columns)
    val_targets_df =pd.DataFrame(val_targets, columns=columns)
    targets_df['chemical formula'] = chemical_formulas
    val_targets_df['chemical formula'] = val_chemical_formulas
    return targets_df, val_targets_df

@st.cache(allow_output_mutation=True)
def feature_dataframe(small_molecules, species):
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=5,
        nmax=5,
        lmax=5,
        average='outer',
        sparse=False,
    )
    feature_vectors = soap.create(small_molecules)
    chemical_formulas = []
    for mol in small_molecules:
        chemical_formulas.append(mol.get_chemical_formula())
    df = pd.DataFrame(feature_vectors)
    df['chemical formula'] = chemical_formulas
    return df

@st.cache(allow_output_mutation=True)
def val_feature_dataframe(small_molecules, species):
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=5,
        nmax=5,
        lmax=5,
        average='outer',
        sparse=False,
    )
    chemical_formulas = []
    for mol in small_molecules:
        chemical_formulas.append(mol.get_chemical_formula())
    feature_vectors = soap.create(small_molecules)
    df = pd.DataFrame(feature_vectors)
    df['chemical formula'] = chemical_formulas
    return df

@st.cache(allow_output_mutation=True)
def initialize():
    geometries = get_geometries()
    all_molecules = molecules_lists(geometries)
    small_molecules, targets, val_small_molecules, val_targets = \
        all_molecules[0],all_molecules[1],all_molecules[2],all_molecules[3]
    targets_df, val_targets_df = get_targets(small_molecules, targets, val_small_molecules, val_targets)
    species = set()
    for mol in small_molecules:
        species.update(mol.get_chemical_symbols())
    for mol in val_small_molecules:
        species.update(mol.get_chemical_symbols())
    df = feature_dataframe(small_molecules, species)
    val_df = val_feature_dataframe(val_small_molecules, species)
    return geometries, small_molecules, targets, val_small_molecules, val_targets, df, val_df, targets_df, val_targets_df

geometries, small_molecules, targets, val_small_molecules, val_targets, df, val_df, targets_df, val_targets_df = initialize()

st.markdown('''
----------------
''')
st.header('1. DATA:')

st.markdown(f"""
Total of {len(small_molecules)+len(val_small_molecules)} organic molecules, seperated in two parts:
* {len(small_molecules)} used as train/validation set. 
* {len(val_small_molecules)} 'unseen' data for testing. 
""")
st.write('''
Source: [Machine learning of molecular electronic properties
in chemical compound space](https://iopscience.iop.org/article/10.1088/1367-2630/15/9/095003)
''')
st.subheader('Molecules')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-06.png', width=40)
st.write('Molecules are described by the chemical formula and the atomic positions') 

if st.button('Show Random Molecule'):
    random_geometry = random.choice(geometries)   
    with open(f'random_geometry.xyz','w') as f:
        f.write(random_geometry)
    random_molecule = io.read(f'random_geometry.xyz')     
    raw_html = f"""
        <h2 align="center">{random_molecule.get_chemical_formula()}</h2>
    """
    st.components.v1.html(raw_html)
    plot_molecule(random_molecule) 
    st.write(pd.DataFrame(random_molecule.get_positions(),index=random_molecule.get_chemical_symbols(),columns=['x','y','z']))

st.write('''
Python package for molecules: [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
''')
st.subheader('Properties')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-09.png', width=50)

def plot_target(this_df, title):
    mean = this_df.mean()
    std = this_df.std()
    st.subheader(title)
    st.write(f"mean: {round(mean,2)}, std: {round(std,2)}")
    fig, ax = plt.subplots()
    ax = sns.histplot(
        this_df,
    )
    st.pyplot(fig)

if st.button('Atomization Energies'):
    plot_target(targets_df['ae_pbe0'],'Atomization Energies (kcal/mol)')

if st.button('Polarizability'):
    plot_target(targets_df['p_pbe0'],'Polarizability (&#8491;&macr;&#179;)')

if st.button('Excitation energies'):
    plot_target(targets_df['lumo_pbe0']-targets_df['homo_pbe0'],'HOMO-LUMO Exictation energies (eV)')


st.write('''
Properties are calculated with [FHIaims code](https://fhi-aims.org/) using the [PBE0](https://en.wikipedia.org/wiki/Hybrid_functional#PBE0)
approximation to [Kohn-Sham Density Functional Theory](https://en.wikipedia.org/wiki/Kohn-Sham_equations).
''')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-10.png', width=70)


st.markdown('''
----------------
''')
st.header('2. FEATURE ENGINEERING')

st.image('Presi Artworks_01/Maria Final Spiced Presi_01-05.png')
st.write('Translation and Rotation invariant representation for ML simulations')
st.subheader('DESCRIPTORS')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-07.png', width=60)

st.write('''
* [Coulomb Matrix](https://singroup.github.io/dscribe/latest/tutorials/descriptors/coulomb_matrix.html)
* [Smooth Overlap of Atomic Positions (SOAP)](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html#smooth-overlap-of-atomic-positions)
''')
st.write('''
Python package for Descriptors: [Dscribe](https://doi.org/10.1016/j.cpc.2019.106949)
''')



if st.button('Apply SOAP'):

    st.write( df.sample(20))



st.markdown('''
----------------
''')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('3. MACHINE LEARNING')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-08.png', width=60)
st.write("""
**Model of choice:** Kernel Ridge Regression 

$\color{blue}{\\text{Pros:}}$ Flexibility, Easy Hyperparameter Tuning
 
$\color{red}{\\text{Cons:}}$ Tendency to Overfit


""")

st.subheader('Create KRR  Prediction Model')
st.subheader('Energy')
if st.button('Polynomial Kernel '):
    st.write("Fitting on train set")
    train_pred_df, test_pred_df, train_target_df, test_target_df = train_test_split(df.drop(columns=['chemical formula']), targets_df['ae_pbe0'], test_size=0.33, random_state=0)
    model = KernelRidge(kernel='poly', alpha=0.00045, degree=2)
    model.fit(train_pred_df, train_target_df)
    filename = 'energy_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write("Calculating predictions on test set")
    predictions = model.predict(test_pred_df)
    rmse = mean_squared_error(test_target_df, predictions, squared=False)
    st.write(f'rmse: {rmse:.2f} kcal/mol or {rmse*0.043:.2f}  (eV)')
    st.write('R2 score is: %.4f' % r2_score(test_target_df, predictions))
    plt.plot(test_target_df, predictions, 'o')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    # We also plot a diagonal line to show where the prediction should end up.
    plt.plot(list(test_target_df),list(test_target_df),'k-') # identity line
    st.pyplot(bbox_inches='tight')

st.subheader('Polarizability')
if st.button('Radial basis function kernel'):
    train_pred_df, test_pred_df, train_target_df, test_target_df = train_test_split(df.drop(columns=['chemical formula']), targets_df['p_pbe0'], test_size=0.33, random_state=0)
    st.write("Fitting on train set")
    model = KernelRidge(kernel='rbf', alpha=3.1622776601683795e-10, gamma=1e-8)
    model.fit(train_pred_df, train_target_df)
    filename = 'polarizattion_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write("Calculating predictions on test set")
    predictions = model.predict(test_pred_df)
    rmse = mean_squared_error(test_target_df, predictions, squared=False)
    st.write(f'rmse: {rmse:.2f} kcal/mol or {rmse*0.043:.2f}  (eV)')
    st.write('R2 score is: %.4f' % r2_score(test_target_df, predictions))
    plt.plot(test_target_df, predictions, 'o')
    plt.xlabel('True Polarizability')
    plt.ylabel('Predicted Polarizability')
    # We also plot a diagonal line to show where the prediction should end up.
    plt.plot(list(test_target_df),list(test_target_df),'k-') # identity line
    st.pyplot(bbox_inches='tight')    



st.subheader('Excitation Energy')
if st.button('Radial basis function kernel '):
    train_pred_df, test_pred_df, train_target_df, test_target_df = train_test_split(df.drop(columns=['chemical formula']), targets_df['lumo_pbe0']-targets_df['homo_pbe0'], test_size=0.33, random_state=0)
    st.write("Fitting on train set")
    model = KernelRidge(kernel='poly', alpha=1.7782794100696326, degree=3)
    model.fit(train_pred_df, train_target_df)
    filename = 'polarizattion_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write("Calculating predictions on test set")
    predictions = model.predict(test_pred_df)
    rmse = mean_squared_error(test_target_df, predictions, squared=False)
    st.write(f'rmse: {rmse:.2f} kcal/mol or {rmse*0.043:.2f}  (eV)')
    st.write('R2 score is: %.4f' % r2_score(test_target_df, predictions))
    plt.plot(test_target_df, predictions, 'o')
    plt.xlabel('True Polarizability')
    plt.ylabel('Predicted Polarizability')
    # We also plot a diagonal line to show where the prediction should end up.
    plt.plot(list(test_target_df),list(test_target_df),'k-') # identity line
    st.pyplot(bbox_inches='tight')  

st.write('Find out more about KKR [here](https://youtu.be/H_MVlljpYHw)')

st.markdown('''
----------------
''')

st.header('4. PREDICT')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-12.png', width=60)


input = st.selectbox('Choose a molecule:', val_df['chemical formula'])
loaded_model_pol = pickle.load(open('polarizattion_model.sav', 'rb'))
loaded_model_en = pickle.load(open('energy_model.sav', 'rb'))
features = val_df[val_df['chemical formula']==input].drop(columns=['chemical formula'])
pol_result = loaded_model_pol.predict(features)
en_result = loaded_model_en.predict(features)
pol_target = val_targets_df[val_targets_df['chemical formula']==input]['p_pbe0']
en_target = val_targets_df[val_targets_df['chemical formula']==input]['ae_pbe0']

st.markdown(f'''
|   {input}    | ML prediction          | Target                 | 
|--------------|------------------------|------------------------|
|Polarizability (&#8491;&macr;&#179;)   |{round(pol_result[0],2)}|{round(list(pol_target)[0],2)}|
|Energy (kcal/mol)|{round(en_result[0],2)}|{round(list(en_target)[0],2)}|
''')

st.markdown('''
----------------
''')
st.markdown("""
#
#
#
#
""")
st.markdown('''
### Appendix
''')
st.markdown('''
----------------
''')

st.header('Kernel Ridge Regression Playground')
st.subheader('Playgound Data')

st.markdown('''
Gaussian function with normal distribution random noise:

    def f(x):
        return np.exp(-x**2)

    x = np.linspace(0,2,30)
    noise = np.random.normal(0,0.3,30)
    target = f(x) + noise

''')


def f1(x):
    return np.exp(-x**2)
x1 = np.linspace(0,2,30)
noise = np.random.normal(0,0.3,30, )
target = f1(x1)+noise
training_data = pd.DataFrame({'x1':x1})
training_target = pd.DataFrame(target)

plt.plot(x1, f1(x1), label="True target", linewidth=2)
plt.scatter(
    training_data,
    training_target,
    color="black",
    label="Noisy target",
)
plt.legend()
plt.xlabel("data")
plt.ylabel("target")
st.pyplot()
st.subheader('Choose Hyperparameters')
kernel = st.selectbox('Choose kernel:', ['Polynomial','Radial basis function kernel'])
alpha = st.number_input('Input alpha', min_value=1e-11, max_value=None)
exp_gamma = st.number_input('Input gamma for RBF', min_value=1e-11, max_value=None)
degree = st.number_input('Input degree for polynomial', min_value=1, max_value=5)

if st.button('Start'):
    if kernel=='polynomial':
        kernel_ridge = KernelRidge(kernel='poly', alpha=alpha, degree=degree)
    elif kernel=='Radial basis function kernel':
        kernel_ridge = KernelRidge(kernel='rbf', alpha=alpha, gamma=exp_gamma)
    kernel_ridge.fit(training_data, training_target)
    predictions = pd.DataFrame(kernel_ridge.predict(training_data))
    plt.plot(x1, f1(x1), label="True target", linewidth=2)
    plt.scatter(
        training_data,
        training_target,
        color="black",
        label="Noisy target",
    )
    plt.scatter(
        training_data,
        predictions,
        color="red",
        label="Prediction",
    )
    plt.legend()
    plt.xlabel("data")
    plt.ylabel("target")
    st.pyplot()



st.write('''
-----

## Quantum Chemistry Model

How long does a (super expensive) Quantum Chemistry calculation take?
''')
if st.button('Start Super Expensive (DFT/PBE0) Quantum Mechanical Calculation'):
    random_molecule = random.choice(val_small_molecules)
    st.write(f"Your calculation has started. It may take several minutes")
    os.system(f'say "Your calculation has started. It may take several minutes"')
    # e_atoms = []
    # for i,atom in enumerate(random_molecule):
    #     atom = Atoms(atom.symbol)
    #     atom.calc = Aims(
    #         xc='pbe0',
    #         species_dir='/Users/maria/FHIaims/species_defaults/defaults_2010/tight',
    #         tier=2,
    #         sc_accuracy_rho=1E-6,
    #         sc_accuracy_eev=1E-6,
    #         sc_accuracy_etot=1E-6,
    #         aims_command='mpirun /Users/maria/FHIaims/build/aims.220309.scalapack.mpi.x',
    #         outfilename='aims.out',
    #         spin='collinear',
    #         default_initial_moment='hund',
    #     )
    #     e_atoms.append(atom.get_potential_energy())

    random_molecule.calc = Aims(
            xc='pbe0',
            species_dir='/Users/maria/FHIaims/species_defaults/defaults_2010/tight',
            tier=2,
            sc_accuracy_rho=1E-6,
            sc_accuracy_eev=1E-6,
            sc_accuracy_etot=1E-6,
            aims_command='mpirun /Users/maria/FHIaims/build/aims.220309.scalapack.mpi.x',
            outfilename='aims.out',
            spin='collinear',
            default_initial_moment=0.001,
        )
    total_energy = random_molecule.get_potential_energy()
    os.system(f'say "Your calculation has finished."')

