from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(
    smi: str, num_confs: int = 10, opt: bool = True, random_seed: int=42, num_threads: int=0, prune_rms_thresh: float=0.5
) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"invalid SMILES: {smi}")
        return None

    mol = Chem.AddHs(mol)
    conf_id = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        pruneRmsThresh=prune_rms_thresh,
        randomSeed=random_seed,
        numThreads=num_threads,
    )
    if not conf_id:
        return None
    if opt:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    return mol


def check_conformers(rdkit_confs: Chem.Mol, target_smi) -> list[bool]:
    output = []
    for idx in range(rdkit_confs.GetNumConformers()):
        cpy = Chem.Mol(rdkit_confs)
        cpy.RemoveAllConformers()
        cpy.AddConformer(rdkit_confs.GetConformer(idx), assignId=True)

        Chem.AssignAtomChiralTagsFromStructure(cpy)
        mol = Chem.RemoveHs(cpy)

        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)
        is_same = smi == target_smi

        smi_can = Chem.MolToSmiles(
            mol,
            isomericSmiles=True,
        )
        is_same_can = smi_can == target_smi

        output.append(is_same or is_same_can)
    return output
