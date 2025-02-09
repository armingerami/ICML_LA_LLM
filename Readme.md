# Toward Making Linear Attention Practical
This is the Github repo for the ICML submission "Toward Making Linear Attention Practical".

## Table of Contents

- [Installation](#installation)
- [Recreate Profiling Experiments](#Recreate\_the\_profiling\_experiments)
- [Recreate LLM Training](#features)
- [Recreate LLM Benchmarks](#contributing)
- [Use it in your project](#license)

## Installation

NOTE: Plaese use "venv" to set up you virtual environment. Do not use "CONDA".<br>
After activating you virtual environment and making sure your GPU is accesible:

```
pip install torch
cd linear_attention
module load gcc
module load cuda
python setup_fastmax.py install
```
The "gcc" version must be between 11.0.0 and 13.0.0, and the cuda version afte 11.0.0. The library will only be installed if there's an Nvidia GPU accesible.

## Recreate Profiling Experiments
After installing our library, simply run `profiling.py` in the `linear_attention` directory. If you're using an SLURM environment, simply run the `run_profiling.sh` script. It will request a GPU and take care of installation.
```
cd linear_attention
sbatch run_profiling.sh
```

## Recreate LLM Training
We use [`LitGPT`](https://github.com/Lightning-AI/litgpt) for our implementation. We have modified the `model.py` file to enable using our linear attention, the `config.py` file to define our LLM, and `pretrain.py` to enable training the Wiki40B dataset.<br>
To install `LitGPT`
```
pip install 'litgpt[all]'
```
To prepare the Wiki40B dataset download the english segment of [Wiki40b](https://huggingface.co/datasets/google/wiki40b), and put it in the `/wiki40b/en` folder. Note that the files in this folder should be `.parquet`. You should then run `parquet_to_jason.py` to reformat the files to `.jason`, which LitGPT uses. Make sure that the paths used in `parquet_to_jason.py` match your directory.
