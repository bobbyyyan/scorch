for i in {1..55}; do python scorch_sparse_autoencoder.py --mode test --dataset mnist | tee -a ~/output_raw/sparse_autoencoder-mnist-scorch.txt; done
for i in {1..55}; do python scorch_sparse_autoencoder.py --mode test --dataset cifar10 | tee -a ~/output_raw/sparse_autoencoder-cifar10-scorch.txt; done
for i in {1..55}; do python scorch_sparse_autoencoder.py --mode test --dataset cifar100 | tee -a ~/output_raw/sparse_autoencoder-cifar100-scorch.txt; done
for i in {1..55}; do python scorch_sparse_autoencoder.py --mode test --dataset celeba | tee -a ~/output_raw/sparse_autoencoder-celeba-scorch.txt; done

for i in {1..55}; do python torch_sparse_autoencoder.py --mode test --dataset mnist | tee -a ~/output_raw/sparse_autoencoder-mnist-torch.txt; done
for i in {1..55}; do python torch_sparse_autoencoder.py --mode test --dataset cifar10 | tee -a ~/output_raw/sparse_autoencoder-cifar10-torch.txt; done
for i in {1..55}; do python torch_sparse_autoencoder.py --mode test --dataset cifar100 | tee -a ~/output_raw/sparse_autoencoder-cifar100-torch.txt; done
for i in {1..55}; do python torch_sparse_autoencoder.py --mode test --dataset celeba | tee -a ~/output_raw/sparse_autoencoder-celeba-torch.txt; done
