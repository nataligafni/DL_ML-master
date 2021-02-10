This repository contains source code for "Multi-task deep learning based CT imaging analysis for COVID-19: Classification and Segmentation".
The scripts are written in Keras library, based on TF 2.X. 
The packges in the environment are listed in the "environment.yml" file. 
Before running the main code you should run the following scripts:

1. path_and_params (folder):
#Configuration() - define the paths in the configuration file
#params() - decide what task you want to start - segmentation or classification - you should define it in: self.data_type ('seg_only' or 'not_seg')
2. utils (folder):
you should run the train_test_split.py script
3. main.py
decide what you want to run - training or testing (comment out the part you don't want - #Train (row 83) or #Test and Evaluation (row 89)
