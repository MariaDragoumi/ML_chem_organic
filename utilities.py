import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dscribe.descriptors import SOAP
from ase import io
from ase.visualize import view


@st.cache(allow_output_mutation=True)
def get_geometries():
    """
    Reads dataset input file.
    Returns a list were each entry is a single molecule geometry.
    """
    with open('gdb7-13/dsgdb7njp.xyz', 'r') as f:
        geometries = f.read()
    geometries = geometries.split('\n\n')
    return geometries


@st.cache(allow_output_mutation=True)
def molecules_lists(geometries):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: list with strings
    Output: lists with ASE objects
    """
    val_set_index = np.random.choice(
        range(len(geometries[1:])),
        size=int(len(geometries)/3),
    )
    small_molecules = []
    targets = []
    val_small_molecules = []
    val_targets = []
    for i, geometry in enumerate(geometries):
        if i not in val_set_index:
            with open(f'geometries/{i}.xyz', 'w') as f:
                f.write(geometry)
            small_molecules.append(io.read(f'geometries/{i}.xyz'))
            targets.append(
                [float(val) for val in geometry.split('\n')[1].split(' ')]
            )
            os.remove(f'geometries/{i}.xyz')
        else:
            with open(f'geometries/{i}.xyz', 'w') as f:
                f.write(geometry)
            val_small_molecules.append(io.read(f'geometries/{i}.xyz'))
            val_targets.append(
                [float(val) for val in geometry.split('\n')[1].split(' ')]
            )
            os.remove(f'geometries/{i}.xyz')
    return small_molecules, targets, val_small_molecules, val_targets


def plot_molecule(molecule):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
    molecule.translate(-molecule.get_center_of_mass())
    html_item = view(molecule, viewer='x3d')
    return html_item._repr_html_()
    


@st.cache(allow_output_mutation=True)
def get_targets(small_molecules, targets, val_small_molecules, val_targets):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
    columns = [
        'ae_pbe0',
        'p_pbe0',
        'p_scs',
        'homo_gw',
        'homo_pbe0',
        'homo_zindo',
        'lumo_gw',
        'lumo_pbe0',
        'lumo_zindo',
        'ip_zindo',
        'ea_zindo',
        'e1_zindo',
        'emax_zindo',
        'imax_zindo',
    ]
    chemical_formulas = []
    for mol in small_molecules:
        chemical_formulas.append(mol.get_chemical_formula())
    val_chemical_formulas = []
    for mol in val_small_molecules:
        val_chemical_formulas.append(mol.get_chemical_formula())
    targets_df = pd.DataFrame(targets, columns=columns)
    val_targets_df = pd.DataFrame(val_targets, columns=columns)
    targets_df['chemical formula'] = chemical_formulas
    val_targets_df['chemical formula'] = val_chemical_formulas
    return targets_df, val_targets_df


@st.cache(allow_output_mutation=True)
def feature_dataframe(small_molecules, species):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
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
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
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
def initialize(geometries):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
    all_molecules = molecules_lists(geometries)
    small_molecules, targets, val_small_molecules, val_targets = \
        all_molecules[0], all_molecules[1], all_molecules[2], all_molecules[3]
    targets_df, val_targets_df = \
        get_targets(small_molecules, targets, val_small_molecules, val_targets)
    species = set()
    for mol in small_molecules:
        species.update(mol.get_chemical_symbols())
    for mol in val_small_molecules:
        species.update(mol.get_chemical_symbols())
    df = feature_dataframe(small_molecules, species)
    val_df = val_feature_dataframe(val_small_molecules, species)
    return small_molecules, targets, val_small_molecules, val_targets, df, val_df, targets_df, val_targets_df


def plot_target(this_df, title):
    """
    Returns a list were each entry is a Molecule (ASE) object.

    Input: None
    Output: lists
    """
    mean = this_df.mean()
    std = this_df.std()
    st.subheader(title)
    st.write(f"mean: {round(mean,2)}, std: {round(std,2)}")
    fig, ax = plt.subplots()
    ax = sns.histplot(
        this_df,
    )
    st.pyplot(fig)
