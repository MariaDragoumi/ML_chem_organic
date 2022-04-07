import random
import pickle
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from ase import io
from ase.calculators.aims import Aims
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from utilities import get_geometries, initialize, plot_molecule, plot_target


# Options for streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


# Intro
st.write('''
# Machine Learning in Chemistry:
## Prediction of organic molecule properties

---
''')
st.image(
    'Presi Artworks_01/star_molecules.webp'
)
st.write('[Image Source](https://www.nature.com/articles/s41570-020-0189-9)')

expander_bar_what = st.expander("What?")
expander_bar_what.image('Presi Artworks_01/Maria Final Spiced Presi_01-01.png')

expander_bar_why = st.expander("Why?")
expander_bar_why.image('Presi Artworks_01/Maria Final Spiced Presi_01-02.png')
expander_bar_why.image('Presi Artworks_01/Maria Final Spiced Presi_01-03.png')

expander_bar_how = st.expander("How?")
expander_bar_how.image('Presi Artworks_01/Maria Final Spiced Presi_01-04.png')


# Collect variables
geometries = get_geometries()
small_molecules, targets,\
    val_small_molecules, val_targets, \
    df, val_df, targets_df, val_targets_df \
    = initialize(geometries)


# Part 1
st.markdown('''
----------------
''')
st.header('1. DATA:')
st.markdown(f"""
Total of {len(small_molecules)+len(val_small_molecules)} organic molecules,
seperated in two parts:
* {len(small_molecules)} used as train/validation set.
* {len(val_small_molecules)} 'unseen' data for testing.
""")
st.write('''
Source: [Machine learning of molecular electronic properties in chemical
compound space]
(https://iopscience.iop.org/article/10.1088/1367-2630/15/9/095003)
''')

# Part 1a
st.subheader('Molecules')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-06.png', width=40)
st.write(
    'Molecules are described by the chemical formula and the atomic positions'
)

if st.button('Show Random Molecule'):
    random_geometry = random.choice(geometries)
    with open('random_geometry.xyz', 'w') as f:
        f.write(random_geometry)
    random_molecule = io.read('random_geometry.xyz')
    raw_html = f"""
        <h2 align="center">{random_molecule.get_chemical_formula()}</h2>
    """
    st.components.v1.html(raw_html)
    raw_html = plot_molecule(random_molecule)
    st.components.v1.html(raw_html, height=400)
    st.write(pd.DataFrame(random_molecule.get_positions(),
             index=random_molecule.get_chemical_symbols(),
             columns=['x', 'y', 'z']))

st.write('''
Python package for molecules: [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
''')

# Part 1b
st.subheader('Properties')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-09.png', width=50)

if st.button('Atomization Energies'):
    plot_target(targets_df['ae_pbe0'], 'Atomization Energies (kcal/mol)')

if st.button('Polarizability'):
    plot_target(targets_df['p_pbe0'], 'Polarizability (&#8491;&macr;&#179;)')

if st.button('Excitation energies'):
    plot_target(targets_df['lumo_pbe0']-targets_df['homo_pbe0'],
                'HOMO-LUMO Exictation energies (eV)')

st.write('''
Properties are calculated with [FHIaims code](https://fhi-aims.org/)
using the [PBE0](https://en.wikipedia.org/wiki/Hybrid_functional#PBE0)
approximation to
[Kohn-Sham Density Functional Theory]
(https://en.wikipedia.org/wiki/Kohn-Sham_equations).
''')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-10.png', width=70)


# Part 2
st.markdown('''
----------------
''')
st.header('2. FEATURE ENGINEERING')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-05.png')
st.write(
    'Translation and Rotation invariant representation for ML simulations'
)
st.subheader('DESCRIPTORS')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-07.png', width=60)
st.write('''
* [Coulomb Matrix]
(https://singroup.github.io/dscribe/latest/tutorials/descriptors/coulomb_matrix.html)
* [Smooth Overlap of Atomic Positions (SOAP)]
(https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html#smooth-overlap-of-atomic-positions)
''')
st.write('''
Python package for Descriptors: [Dscribe]
(https://doi.org/10.1016/j.cpc.2019.106949)
''')
if st.button('Apply SOAP'):

    st.write(df.sample(20))


# Part 3
st.markdown('''
----------------
''')
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
    train_pred_df, test_pred_df, train_target_df, test_target_df \
        = train_test_split(
            df.drop(columns=['chemical formula']),
            targets_df['ae_pbe0'],
            test_size=0.33,
            random_state=0,
        )
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
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot(list(test_target_df), list(test_target_df), 'k-')
    st.pyplot(bbox_inches='tight')

st.subheader('Polarizability')
if st.button('Radial basis function kernel'):
    train_pred_df, test_pred_df, train_target_df, test_target_df \
        = train_test_split(
            df.drop(columns=['chemical formula']),
            targets_df['p_pbe0'],
            test_size=0.33,
            random_state=0,
        )
    st.write("Fitting on train set")
    model = KernelRidge(kernel='rbf', alpha=3.1622776601683795e-10, gamma=1e-8)
    model.fit(train_pred_df, train_target_df)
    filename = 'polarization_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write("Calculating predictions on test set")
    predictions = model.predict(test_pred_df)
    rmse = mean_squared_error(test_target_df, predictions, squared=False)
    st.write(f'rmse: {rmse:.2f} kcal/mol or {rmse*0.043:.2f}  (eV)')
    st.write('R2 score is: %.4f' % r2_score(test_target_df, predictions))
    plt.plot(test_target_df, predictions, 'o')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot(list(test_target_df), list(test_target_df), 'k-')
    st.pyplot(bbox_inches='tight')

st.subheader('Excitation Energy')
if st.button('Radial basis function kernel '):
    train_pred_df, test_pred_df, train_target_df, test_target_df \
        = train_test_split(
            df.drop(columns=['chemical formula']),
            targets_df['lumo_pbe0']-targets_df['homo_pbe0'],
            test_size=0.33,
            random_state=0
        )
    st.write("Fitting on train set")
    model = KernelRidge(kernel='poly', alpha=1.7782794100696326, degree=3)
    model.fit(train_pred_df, train_target_df)
    filename = 'excitation_energy_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write("Calculating predictions on test set")
    predictions = model.predict(test_pred_df)
    rmse = mean_squared_error(test_target_df, predictions, squared=False)
    st.write(f'rmse: {rmse:.2f} kcal/mol or {rmse*0.043:.2f}  (eV)')
    st.write('R2 score is: %.4f' % r2_score(test_target_df, predictions))
    plt.plot(test_target_df, predictions, 'o')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot(list(test_target_df), list(test_target_df), 'k-')
    st.pyplot(bbox_inches='tight')

st.write('Find out more about KKR [here](https://youtu.be/H_MVlljpYHw)')

# Part 4
st.markdown('''
----------------
''')
st.header('4. PREDICT')
st.image('Presi Artworks_01/Maria Final Spiced Presi_01-12.png', width=60)

input = st.selectbox('Choose a molecule:', val_df['chemical formula'])
loaded_model_pol = pickle.load(open('polarization_model.sav', 'rb'))
loaded_model_en = pickle.load(open('energy_model.sav', 'rb'))
features = val_df[val_df['chemical formula'] == input]
features.drop(columns=['chemical formula'], inplace=True)
pol_result = loaded_model_pol.predict(features)
en_result = loaded_model_en.predict(features)
target = val_targets_df[val_targets_df['chemical formula'] == input]
pol_target = target['p_pbe0']
en_target = target['ae_pbe0']

st.markdown(f'''
|   {input}    | ML prediction          | Target                 |
|--------------|------------------------|------------------------|
|Polarizability (&#8491;&macr;&#179;)   |{round(pol_result[0],2)}|{round(list(pol_target)[0],2)}|
|Energy (kcal/mol)|{round(en_result[0],2)}|{round(list(en_target)[0],2)}|
''')

# End of main part
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


def f(x):
    return np.exp(-x**2)
x = np.linspace(0, 2, 30)
noise = np.random.normal(0, 0.3, 30)
target = f(x)+noise
training_data = pd.DataFrame({'x': x})
training_target = pd.DataFrame(target)

plt.plot(x, f(x), label="True target", linewidth=2)
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
kernel = st.selectbox('Choose kernel:',
                      ['Polynomial', 'Radial basis function kernel'])
alpha = st.number_input('Input alpha',
                        min_value=1e-11, max_value=None)
exp_gamma = st.number_input('Input gamma for RBF',
                            min_value=1e-11, max_value=None)
degree = st.number_input('Input degree for polynomial',
                         min_value=1, max_value=5)

if st.button('Start'):
    if kernel == 'Polynomial':
        kernel_ridge = KernelRidge(kernel='poly', alpha=alpha, degree=degree)
    elif kernel == 'Radial basis function kernel':
        kernel_ridge = KernelRidge(kernel='rbf', alpha=alpha, gamma=exp_gamma)
    kernel_ridge.fit(training_data, training_target)
    predictions = pd.DataFrame(kernel_ridge.predict(training_data))
    plt.plot(x, f(x), label="True target", linewidth=2)
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
if st.button('Start DFT/PBE0 Calculation'):
    random_molecule = random.choice(val_small_molecules)
    st.write("Your calculation has started. It may take several minutes")
    os.system('say "Your calculation has started. It may take several minutes"')
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
            default_initial_moment=0.01,
        )
    total_energy = random_molecule.get_potential_energy()
    os.system('say "Your calculation has finished."')
