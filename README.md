# Fundus Images
This repository provides the pipeline to reproduce the results from [In-Depth Exploration of Deep Aging Clocks in Ophthalmology Using the AEyeDB Dataset](TODO). 


## Data 
For the Analysis we used the [Papila](https://www.nature.com/articles/s41597-022-01388-1), [OIA-ODIR](https://link.springer.com/chapter/10.1007/978-3-030-71058-3_11) and [AEyeDB](TODO) datasets. In order to obtain the AEyeDB dataset, please submit a reasonable research request to the principal investigator [Professor Dr. Dominik Heider](https://heiderlab.de/).

## Preprocessing
The functions used for the preprocessing steps can be found in the Preproc.py file. During preprocessing, images were cropped, downsized, equalized, and rotated. 

<p float="left">
  <img src="res/t0.png" alt="original_image" width="400"/>
  <img src="res/t1.png" alt="cropped_image" width="200"/>
  <img src="res/t3.png" alt="equalized_image" width="200"/>
  <em>The left image is the original one, in the middle is cropped and downsized to 224x224 image, and the right image is equalized and randomely rotated.  </em>
</p>

For the color histogram equalization we used a cutoff color intensity of 20, for more information see [In-Depth Exploration of Deep Aging Clocks in Ophthalmology Using the AEyeDB Dataset](TODO).

## Learning Pipelin
We trained the models on the HPC HHU Hilbert. The pipeline we used is documented in the main.py file. For the training we used following hyperparameters:
   * Learning Rate: 0.0001 (ResNet152), 0.001 (DenseNet201, Inception-v3)
   * Weight Decay: 0.0001
   * Bacht Size: 50
   * Number of GPUs: 5
   * Number of CPUs: 4
   
# Publications
Please cite following publication if you use the results:

# Authors
   * Jan Benedikt Ruhland, jan.ruhland@hhu.de
   * Iraj Masoudian
   * Doguhan Bahcivan
   * Daniel Tiet
   * Erik Yi-Li Sun Gal
   * Hung Luu
   * Prof. Dr. Dominik Heider, dominik.heider@hhu.de, principal inversitgator
   
   
# License
MIT license (see license file). 

# Acknowledgments
This work is partially funded by hessian.AI in the Connectom project VirtualDoc.
We also wish to thank the students of Heinrich-Heine-University who participated in the study. 
