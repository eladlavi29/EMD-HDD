# HYPERBOLIC DISTANCE BASED ON EMD AND DIFFUSION DESIGNED FOR HYPERSPECTRAL IMAGING

## Prerequisites

* [Numpy](https://numpy.org/install/)
* [PyTorch](https://pytorch.org/)

## Usage Example 

```python
# X is a tensor of an image by its original shape
# y is the pixel's ground truth (number), where 0 represents an unlabeled pixel
# factor_w and factor_h are the dimensions of non-overlapping patches for the method
train_acc,test_acc, _,_ = wasser_hdd(X,y, factor_w, factor_h))
```
----------------------------------------------------------------------------

## Consts and Hyper parameters
utils/consts.py includes a list of all of the constants and hyper-parameters we use. More Specifically, here is a list of all of the constants that you might want to change:

METRIC_BANDS = 'euclidean' meaning: determines the metric that is used to calculate the distance between the bands. possible values: 'euclidean' or 'cosine'.

METRIC_PIXELS = 'euclidean' meaning: determines the metric that is used to calculate the distance between the pixels. possible values: 'euclidean' or 'cosine'.

USAGE: if you want to change a constant you can either change it in consts.py directly, or in the python script you are running, for example: in run_test.py: consts.METRIC_PIXELS = ‘cosine’.

HIERARCHICAL_METHOD = 'HDD' meaning: it is a constant, which you could set to any value you would like for experiments on based metric for the pipeline which are not HDD.
