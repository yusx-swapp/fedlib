# RaFFM: Resource-aware Federated Learning

**IMPORTANT**

This repository is for the reproducibility for under reviewed paper, and this repository is running in a single device.

If you desired deploy the proposed method to a real cross-silo FL enriroment. Please mimic the usage of the package in `utils` with provides scripts in your edge environments.


## Dependencies

To reproduce our experiments, please follow the instructions to set up environments:

First, creating a python environment via conda:

```
conda create -n raffm python=3

#activate the environment
source activate raffm
```

Then, installing requirement python packages:

[Pytorch](https://pytorch.org/get-started/locally/) installation (for Linux):

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

Next, installing other python package dependencies:

```
cd RaFFM/
pip install -r requirements.text 
```

You may further need run:
```
pip install accelerate -U
```


We are all set about software dependencies, to make sure you can running our codes, you may further need a GPU.
<!-- If you are familiar with Docker, please follow the instructions in the [Docker/](../../Docker/README.md), and build a docker image to run our experiments. -->

## Reproduce experiments
To quick explore our experiment on GLUE Benchmark, you can simple follow the instruction:
```
chmod +x run.sh
./run_experiment.sh --dataset [dataset name]
```

For instance:
```
./run_experiment.sh --dataset sst2
./run_experiment.sh --dataset stsb
```
It will automatically excute RoBERTa with 100 clients and 10% random clients participation each communication round.


Alternatively, you can also run via python scripts, and explore all the arguments by yourself.

Here is an example:
```
python fl_glue_spp.py --eval_lm --spp --algo raffm --split_data --num_clients 100 --num_rounds 50 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --model bert-large --log_dir log_glue/bertlarge/sst2
```