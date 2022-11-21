## Vision Transformers: Comprehension and Implementation
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

#### Author:
* [Pume Tuchinda]

### Overview
This repository contains an implmentation of the Vision Transformer, as desribed in the ICLR2021 Paper An Image is Worth 16x16 Words: Transformers For Image Recognition at Scale (https://arxiv.org/pdf/2010.11929.pdf) in PyTorch. The original repository of the code (implemented in JAX) is available at: https://github.com/google-research/vision_transformer.

### Implementation
The implementation can be summed up in `train.py` and `vit/model.py`, where you will find the model implementation entirely from scratch in PyTorch and the training script for the MNIST dataset. To run the training loop a simple `python train.py` will be able to run the 5 epochs, but if you wish to run for longer you would just have to change the parameter at line 34 to the number of epochs desired in `train.py` script. 
