Repository for ICLR submission "ConforFormer"
=============================================

## Navigation
 | Folder | Inshort | Description |
 |--------|---------|-------------|
 | [unimol](./unimol/) | New additions to Uni-Mol | This is a fork of the origional repository with the additions and alterations made during the course of this research. Primary changes include the addition of models, tasks, and losses for contrastive learning as well as new dataloading to direct use with OpenMolecules |
 | [data_processing](./data_processing/) | Scripts for dataset generation | These scripts are for the replication of data(set) generation done for the paper. This includes the reduced Uni-Mol dataset, the filering of OMol for the conformer subset, and the generation of SQLite databases of embeddings and similairty scores necessary for the scripts outlined in [anaylsis](./analysis/) |
 | [Exampling scripts](./example_scripts/) | Bash scripts | These are example script for running finetuning, training a new model, and the inference for th econtrastive benchmark. |
 | [Data Analysis](./analysis/) | Data analysis scripts | Scripts for the Precision-recall curves and marking molecules as special (enantiomers or optisomers) |

Dataset, as generated using the given scripts, will be provided once review has concluded in a central data repository.
For model parameters please visit [the HuggingFace repository](https://huggingface.co/ConforFormer/ConforFormer) for the project.

## Requirements
 For the general requirements see the [origional Uni-Mol repository](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol).
 Additional packages include:
 - [Fairchem](https://github.com/facebookresearch/fairchem): For training directly on OpenMolecules or for its conformer subset.
 - [OpenBabel](https://openbabel.org/index.html): For the generation of the contrastive benchmark or Filtering of OpenMolecules

## Liscensing
 The origional contributions to this work are lesencsed under MIT. For contact please email conforformerreview@gmail.com