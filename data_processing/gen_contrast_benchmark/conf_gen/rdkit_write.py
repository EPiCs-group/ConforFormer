from rdkit import Chem
import numpy as np
import os


def write_xyz_rdkit_confs(rdkit_conf: Chem.Mol, path_to_outdir: os.PathLike, file_name: str):
    """
    Writes the rdkit things into a file.

    args:
      rdkit_conf: whatever rdkit.AllChem.EmbedMultipleConfs outputs
    """
    if not os.path.exists(path_to_outdir):
        os.mkdir(path_to_outdir)

    for i in range(rdkit_conf.GetNumConformers()):
        mol_name = f"{file_name}_{i}.xyz"
        file_path = os.path.join(path_to_outdir, mol_name)
        Chem.rdmolfiles.MolToXYZFile(rdkit_conf, file_path, confId=i)


def write_to_np_array(rdkit_conf: Chem.Mol):
    output_atoms = [atom.GetSymbol() for atom in rdkit_conf.GetAtoms()]

    output_coords = []
    for i in range(rdkit_conf.GetNumConformers()):
        conf = rdkit_conf.GetConformer(i)
        coords = np.array(conf.GetPositions())
        output_coords.append(coords)
    return output_atoms, output_coords

