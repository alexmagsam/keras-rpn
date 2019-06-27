# keras-rpn
Region Proposal Network implementation for object detection using Keras and TensorFlow

This repository was heavily based on the Mask-RCNN repository (https://github.com/matterport/Mask_RCNN).

# Getting Started
## Dependencies
This was developed using Windows 10 Pro with the following dependencies:
* Keras 2.2.4
* tensorflow-gpu 1.12.0
* numpy 1.15.4
* matplotlib 3.0.2
* CUDA 9.0
* cuDNN v6.0
* Python 3.6.5

Other versions of these dependencies are not guaranteed to work.

## Example Scripts
### 2018 Data Science Bowl
1. Download the data from the nuclei dataset from https://www.kaggle.com/c/data-science-bowl-2018/data and extract 
to a folder of your choice.
2. Change the ```stage1_train``` folder name to ```train```.
3. Create a new folder ```validation``` and drag however many samples you'd like for validation 
from ```train``` into ```validation```.  
4. Change the TRAIN_PATH and VALIDATION_PATH variables in the **nucleus/train.py** script 
to where the ```train``` and ```validation``` folders are located.
5. Run ```python nucleus/train.py``` from the command line.





