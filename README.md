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
Main code for training / fine-tuning both the GAN and the discriminator-classifier can be found in the file train.py. 
Run ```python train.py```

### Folder Structure 
```
project
│   README.md
│
└───images
│   │ bathtub.png
│   │ ... 
│   │
│   └───plots
│       │ gen_disc_loss.png
│       │  ...       
│            
└───models
│   │
│   └───plots
│   │   │ accuracy.png
│   │   │ ...
│   │    
│   └───test_2
│       │ 
│       │
│       └───epoch_images
│           │ image_1_at_epoch_005.png
│           │ ...
│ 
└───classifier_summary.txt
│
└───loss_plot_test_1.npz
│
└───loss_plot_test_2.npz
│
└───model_summary.txt
│
└───modelnet10.npz
│   
└───test.py
│    
└───train.py
```


