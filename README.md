## Overview

This work contains the PyTorch implementation of and demonstrations of [NeurIPS 2021: Self-Interpretable Model with Transformation Equivariant Interpretation](https://proceedings.neurips.cc/paper/2021/file/1387a00f03b4b423e63127b08c261bdc-Paper.pdf) (**SITE**)

**Method**
SITE trains a self-interpretable model that offers both consistent predictions and explanations across geometric transformations. This is achieved through the regularization of a self-interpretable module, thereby increasing the model's trustworthiness.
![alt text](https://github.com/yipei-wang/Images/blob/main/SITE/SITE_overview.png)


For academic usage, please consider citing:
<pre>
  @article{wang2021self,
    title={Self-interpretable model with transformation equivariant interpretation},
    author={Wang, Yipei and Wang, Xiaoqian},
    journal={Advances in Neural Information Processing Systems},
    volume={34},
    pages={2359--2372},
    year={2021}
  }
</pre>

## Contents

**Training** notebooks demonstrate the training process of SITE on [MNIST](MNIST_train.ipynb) and [CIFAR](CIFAR_train.ipynb) datasets.

**Example** notebooks demonstrate how SITE is used to generate explanations for [MNIST](example_MNIST.ipynb) and [CIFAR](example_CIFAR.ipynb) datasets



  
