# PathFormer: Single Pass Biodegradation Pathway Prediction Utilising Mass Spectra Information
PathFormer is a graph-generation transformer model that predicts complete biodegradation pathways in a single pass, conditioned on both molecular structure and mass spectrometry data.
PathFormer formulates the problem as a graph generation task, capturing the inherent structure of biodegradation pathways and enabling the model to retain awareness of its previous predictions

## Download Datasets
The synthetic pathway USPTO dataset can be downloaded [here](https://www.dropbox.com/scl/fi/ztp4vyqtyxcgo7k5x9kd6/uspto_no_metal_nx_pathways.sqlite?rlkey=lrm8cpu7lndizdtrbd97zi42x&st=1e1su3yf&dl=0).

The enviPath datasets can be downloaded using the EnviPathDownloader.py script.
You will also need to download the spectra file available [here](https://www.dropbox.com/scl/fi/xwjaboc07vnl7aum1u24g/cfmid_spectra.h5?rlkey=in8s2yrr9762cu37rvphg2v8t&st=rp6wg4b8&dl=0) 
and a support file [here](https://www.dropbox.com/scl/fi/tfpz510dx4f91jq2qxeqz/all_uncharged_maps.json?rlkey=2gbhxg7w4ndjh44559kwpjbeu&st=mzi1rj84&dl=0). Place them in the data directory.

## Train Models
To pretrain the PathFormer model on the USPTO dataset, run TrainPathFormer.py with the following command:
`python TrainPathFormer.py PathFormer uspto_no_metal`

To refine on biodegradation datasets, run RefinePathFormer.py with the following command:
`python TrainPathFormer.py PathFormer envipath_no_metal --weights-dataset results/BinnedPathFormer/uspto_no_metal_8_256_8_roots_spec_binned_mass_1 -tm`