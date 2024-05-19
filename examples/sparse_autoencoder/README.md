# Sparse Autoencoders

## Training

```shell
python torch_sparse_autoencoder.py --mode train --dataset <dataset_name>
```

The options for `dataset_name` are `mnist`, `cifar10`, `cifar100`, and `celeba`.

## Inference

```shell
python torch_sparse_autoencoder.py --mode test --dataset <dataset_name>
python scorch_sparse_autoencoder.py --mode test --dataset <dataset_name>
```
