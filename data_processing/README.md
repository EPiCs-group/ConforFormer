Data processing
===============

 These are the dataprocessing scripts necessary to reproduce the datasets for training and validation to get the main results of the paper.
 For large pipelines, the scripts are numbered in the order of necessary execution. Any additional files, beyond those generated or datasets which need to be downlaoded are provided.

 | Folder | In short | Description |
 |--------|----------|-------------|
 | Reduced dataset | Parition of Uni-Mol dataset | This splits the origional UniMol dataset into 8 even parts. For the paper partition "1" was used. |
 | OMol filtering | Identify and group conformers of OpenMolecules | Pipeline used to identify, group, and organize the OpenMolceuls dataset in the same way UniMol has. |
 | Isomer percent | Identifying isomer in UniMol dataset | A pipeline for identifying isomers within the Uni-Mol dataset. This includes generating a look-up table for efficient training and altering the origional dataset. |
 | Contrastive benchmark | Generation of PharmaIsomer | Pipeline for generating the contrastive benchmark proposed in the paper. |
 
 ## Additional requirements
  The download of the following datasets are necessary:
   - Uni-Mol train and validation: for Reducing dataset and Isomer percent.
   - OpenMolecules: For OMol filtering.
   - ZINC20: For PharmaIsomer.