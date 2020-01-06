# large-lr-experiments
Implements the experiments of the following paper: https://arxiv.org/abs/1907.04595. 

To run the patch experiment, use the command 

```python train_patches.py --data_dir <location of CIFAR10 data> --save_dir <where to save the run>```

This will train 3 models on a modified CIFAR10 dataset (modified according to the patch experiment in our paper): a large initial LR model, a small LR model, and a model which uses activation noise. 

The command 

```python train_cifar.py``` 

will train a model on the standard CIFAR10 dataset and allows specifying the level of activation noise, learning rate, dropout, etc. 
