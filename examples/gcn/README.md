# Graph Convolutional Networks

## Training

```shell
python pyg_gcn.py --mode train --dataset <dataset_name>
```

Options for `dataset_name` are `cora`, `citeseer`, `pubmed`, and `ogbn-arxiv`.

## Inference

```shell
python torch_gcn.py --mode test --dataset <dataset_name>
python scorch_gcn.py --mode test --dataset <dataset_name>
python pyg_gcn.py --mode test --dataset <dataset_name>
python dgl_gcn.py --mode test --dataset <dataset_name>
```
