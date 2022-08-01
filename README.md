# Requirements

## Installed Packages

For Scrips the following packages are required.

```
matplotlib==3.2.2
numpy==1.21.6
opencv-python==4.6.0.66
pandas==1.3.5
scikit-learn==1.0.2
scipy==1.7.3
seaborn==0.11.2
torch==1.12.0
torch==1.12.0
torchvision==0.13.0
tqdm==2.2.3
```



# This Repository contains several files:
1)  Run Python RandomDataGeneration.py to create a dataset with 3 different labels ('triangle', 'circle', 'rectangle') with 64x64 size of images.
2)  NeuralNet.py contains CNN architecture with torch library.
3)  Training from Scratch use Train.py file.
4)  Evaluation.py for testing the data set this file also gives you a confusion matrix evaluation in the "results" folder.
5)  folder 'runs' contains tensorboard loader. use bellow command for tensorboard loader to the terminal.
  ```
  "tensorboard --logdir runs/shape"
```
