# PathFormer: Single Pass Biodegradation Pathway Prediction Utilising Mass Spectra Information
PathFormer is a graph-generation transformer model that predicts complete biodegradation pathways in a single pass, conditioned on both molecular structure and mass spectrometry data.
PathFormer formulates the problem as a graph generation task, capturing the inherent structure of biodegradation pathways and enabling the model to retain awareness of its previous predictions

## Download Datasets
The USPTO dataset can be downloaded from (uh where do I put a ~30GB file)

The enviPath datasets can be downloaded using the EnviPathDownloader.py script.

## Train Models
To pretrain the PathFormer model on the USPTO dataset, run TrainPathFormer.py with the following command:
`python TrainPathFormer.py PathFormer uspto_no_metal`

To refine on biodegradation datasets, run RefinePathFormer.py with the following command:
`python TrainPathFormer.py PathFormer envipath_no_metal --weights-dataset results/BinnedPathFormer/uspto_no_metal_8_256_8_roots_spec_binned_mass_1 -tm`