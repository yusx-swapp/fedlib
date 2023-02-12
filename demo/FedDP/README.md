# Heterogeneous Federated Learning using Dynamic Model Pruning and Adaptive Gradient

This is the implementation for the paper: **Heterogeneous Federated Learning using Dynamic Model Pruning and Adaptive Gradient**

## Dependencies

To reproduce our experiments, please follow the instructions to set up environments:

First, creating a python environment via conda:

```
conda creat -n fedlib python=3

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

Finally, install the FedLib:

```
cd fedlib_root_dir
pip install .  
```

We are all set about software dependencies, to make sure you can running our codes, you may further need a GPU.
If you are familiar with Docker, please follow the instructions in the `Docker/`, and build a docker image to run our experiments.

## Reproduce experiments

We provided user-friendly interface to run our experiments and extend our method to customized applications.

All our parameters are stored in a `*.ini` files under the folder [./config/config.ini](config/config.ini)

We have provide step-by-step instruction on [demo.ipynb](demo.ipynb) for simply run FedDP and how to use the `FedLib`.

If you want to further reproduce our experiments, for instance, FedDF on ResNet-20 and CIFAR-100 with 100 clients and 10% random client participation rate in each round, simply exceute the following command:

### Run experiments on other dataset

We defaultly provide experiments running option on CIFAR-10/100. If you want to repreduce our experiments on Tiny ImageNet or CINIC-10, please download it first, then change the argument `datadir` in `*.yaml` file.

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
```

## License

MIT License
