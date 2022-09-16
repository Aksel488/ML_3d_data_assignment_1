# 3D Generative-Adversarial Modeling Assignment 1

This is an implementation of the 3D-GAN proposed by [https://arxiv.org/abs/1610.07584]. 

### Environment
Create a new python 3.9.13 environment and activate it.

```
conda create -n assignment_1 python=3.9.13
conda activate assignment_1
```

Then install the rest of the packages.

```
conda install -c tensorflow
conda install -c numpy
conda install -c matplotlib
conda install -c time
conda install -c os
```

### Running
Main code for training / fine-tuning both the GAN and the discriminator-classifier can be found in the file main.py. 
Run ```python main.py```

### Folder Structure 
```
project
│   README.md
│
└───images
│   │ bathtub.png
│   │ ...      
│            
└───models
│   │ model_summary.txt
│   │ classifier_summary.txt
│   │
│   └───plots
│   │   │ classifier_accuracy.png
│   │   │ classifier_loss.png
│   │   │ model_1_loss_plot.png
│   │   │ model_2_loss_plot.png
│   │    
│   └───model_1
│   │   │
│   │   └───epoch_images
│   │       │ image_1_at_epoch_010.png
│   │       │ ...
│   │
│   └───model_2
│       │
│       └───epoch_images
│           │ image_1_at_epoch_005.png
│           │ ...
│
└───modelnet10.npz
│   
└───config.py
│    
└───main.py
│    
└───model.py
```


