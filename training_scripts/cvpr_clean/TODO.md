# Evaluation:

## overall goal: unified evaluation using itk interface

Add correct knee flipping to ./OAI\_eval.py

Create lung evaluation script (note notebook) based on https://colab.research.google.com/github/uncbiag/ICON/blob/master/notebooks/ICON_lung_demo.ipynb

Create brain evaluation script (not notebook) based on ../../notebooks/brain_ants_comparison.ipynb

# Data preparation:

## overall goal: 

put scripts for turning folders of medical images into torch.load-able tensors into this directory. Document how to get folders of medical images


# Training:

# overall goal: unified training 

each task will use model def in cvpr_network.py, training process in ../../src/icon_registration/train.py:train_batchfunction

write data loader "batch function" for each dataset

example scripts (OAI) 
```
../gradICON/gradicon_knee_halfres_new.py
../gradICON/gradicon_knee_halfres_new_2ndhalfres.py
```
(but use cvpr_network instead of defiing model in the training script)

need to add optional validation to train_batchfunction.

need to decide where to put two step training logic.


