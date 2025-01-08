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
