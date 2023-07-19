# FedLib: Implementation for FedAvg

This is the implementation for the paper: **Communication-Efficient Learning of Deep Networks from Decentralized Data**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7633920.svg)](https://doi.org/10.5281/zenodo.7633920)

## Dependencies

To reproduce our experiments, please follow the instructions to set up environments:

First, creating a python environment via conda:

```
conda create -n fedlib python=3

#activate the environment
source activate fedlib
```

Then, installing requirement python packages:

[Pytorch](https://pytorch.org/get-started/locally/) installation (for Linux):

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Next, installing other python package dependencies:

```
cd fedlib_root_dir
pip install -r requirements.text 
```

Finally, install the FedLib (if you dont know what does the `pip install .` do, please read the [link](https://stackoverflow.com/questions/39023758/what-does-pip-install-dot-mean)):

```
cd fedlib_root_dir
pip install .  
```
In the above command, we installed our local package `fedlib`.

We are all set about software dependencies, to make sure you can running our codes, you may further need a GPU.
If you are familiar with Docker, please follow the instructions in the [Docker/](../../Docker/README.md), and build a docker image to run our experiments.

## Reproduce experiments

We provided user-friendly interface to run our experiments and extend our method to customized applications.

All our parameters are stored in a `*.ini` files under the folder [./config/config.ini](config/config.ini).
To run our experiment on CIFAR-10 under Non-IID setting and 100 clients with 10% random participate rate, you could run the following command in your terminal:

```
python eval.py --cf config/config.ini
```

You could change the parameters in [./config/config.ini](config/config.ini) to reproduce more our experiments, for instance, you could change the `pruning_threshold = 1e-2` to try different dynamic pruning rate.

The running results would output to your terminal via logger object.

## Run demo via Jupyter Notebook

We have provide step-by-step instruction on [demo.ipynb](demo.ipynb) for simply run FedDP and how to use the `FedLib`.

### Run experiments on other dataset

We defaultly provide experiments running option on CIFAR-10/100. If you want to repreduce our experiments on Tiny ImageNet or CINIC-10, please download it first, then write your `data root path` to the argument `datadir` in `*.ini` file.

## Acknowledgment

Authored by Phd Student Sixing Yu at SwAPP Lab, Department of Computer Science, Iowa State University.

If you are using our work please cite:

```
@misc{yu2023feddp,
  doi = {10.48550/ARXIV.2106.06921},
  url = {https://arxiv.org/abs/2106.06921},
  author = {Yu, Sixing and Nguyen, Phuong and Anwar, Ali and Jannesari, Ali},  
  title = {Heterogeneous Federated Learning using Dynamic Model Pruning and Adaptive Gradient},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
@InProceedings {yu2021spatl,
title = {SPATL: Salient Parameter Aggregation and Transfer Learning for Heterogeneous Federated Learning},
author = {S. Yu and P. Nguyen and W. Abebe and W. Qian and A. Anwar and A. Jannesari},
booktitle = {2022 SC22: International Conference for High Performance Computing, Networking, Storage and Analysis (SC) (SC)},
year = {2022},
issn = {2167-4337},
pages = {495-508},
keywords = {federated learning;heterogeneous system;machine learning;ml;fl},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {nov}
}

```

## License

MIT License
