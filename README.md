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

## Experiments

For experiments, download the data below and run EMD_HDD experiments using the run_test.py script.
After downloading a dataset, put the dataset and the ground truth in the /dataset repository. Then edit lines 52 to 55 in run_test.py accordingly. 
For your convenience, run_test.py is configured for experiments over the PaviaU dataset.

### Datasets

* [Pavia University (PaviaU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) [2]
* [Pavia Centre](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) [2]
* [Botswana](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) [3]
* [Kennedy Space Center (KSC)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) [1]

### Refrences

[1] D. Lunga, S. Prasad, M. M. Crawford, and O. Ersoy,
“Manifold-learning-based feature extraction for classification
of hyperspectral data: A review of advances in manifold learning,” IEEE Signal Processing Magazine, vol. 31, no. 1, pp. 55–
66, 2014.

[2] X. Huang and L. Zhang, “A comparative study of spatial approaches for urban mapping using hyperspectral rosis images
over pavia city, northern italy,” International Journal of Remote Sensing, vol. 30, no. 12, p. 3205–3221, 2009.

[3] Y. Li, J. Wang, T. Gao, Q. Sun, L. Zhang, and M. Tang, “Adoption of machine learning in intelligent terrain classification of
hyperspectral remote sensing images,” Computational Intelligence and Neuroscience, vol. 2020, no. 1, p. 8886932, 2020.
