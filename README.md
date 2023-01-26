# MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods

![MGGNN](https://user-images.githubusercontent.com/71892896/212237134-c3e34fb6-e92a-4d5a-a1cf-541f2c29340e.png)

## Installation

This code requires at least Python 3.8 to run, along with the CPU versions of PyTorch/PyTorch Geometric.  (CUDA versions can be used but are not strictly necessary)

A full list of the required Python libraries is included in `requirements.txt`. 

## Generate Data

Use `utils/gen_data.py` to generate the training and test data with the following command:

```
python utils/gen_data.py --directory='Data/train_grids'
python utils/gen_data.py --directory='Data/test_grids'

```

A full list of arguments can be found by running with `python utils/gen_data.py --help`. 


## Training

Use `train.py` to train the model:
```
python train.py
```
A full list of arguments can be found by running with `python train.py --help`.  A pre-trained model is also included as `Model/model_trained.pth`.

## Testing

Run the evaluation script with
```
python test.py
```
A full list of arguments can be found by running with `python test.py --help`.
