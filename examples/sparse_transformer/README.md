# Sparse Transformer (BigBird)

## Training

```shell
python torch_sparse_transformer.py --mode train --dataset <dataset_name>
```

The options for `dataset_name` are `yahoo_answers`, `imdb`, and `ag_news`.

## Inference

```shell
python torch_sparse_transformer.py --mode test --dataset <dataset_name>
python scorch_sparse_transformer.py --mode test --dataset <dataset_name>
```
