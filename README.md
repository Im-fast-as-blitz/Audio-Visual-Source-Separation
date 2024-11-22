# Source separation projects




## About

This repository contains a implementation on Conv-Tasnet and RTFS models for speach separation.

## Installation


0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

In case you want to use RTFS model, get video embeddings at first (see chapter `How to get video embeddings`). 
## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

For example to train conv_tasnet:
```bash
python3 train.py -cn=conv_tasnet
```

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

To evaluate metrics use trainer/evaluater.py

## How to get video embeddings:

Firstly download video encoder weights by using special script: 
```python get_model_weight.py```

Than get necessary libs and run model inference:

```

cd src/mouth_processor/Lipreading_using_Temporal_Convolutional_Networks

pip install -r requirements.txt

# add paths to model, data directory, and directory to save embeds  
python python main.py --modality video \
                --extract-feats \
                --config-path configs/lrw_resnet18_dctcn.json \ 
                --model-path ../../utils/mouth_model.pth \
                --mouth-patch-path ../../../dla_dataset/mouths \
                --mouth-embedding-out-path ../../../dla_dataset/mouths_embeds \
                --batch-size 1
```


## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
