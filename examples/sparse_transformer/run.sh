# export CUDA_VISIBLE_DEVICES=""
# for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset yahoo_answers | tee -a /scratch/bobbyy/output_raw/bigbird-yahoo_answers-torch.txt; done
# for i in {1..15}; do python scorch_sparse_transformer.py --mode test --dataset yahoo_answers | tee -a /scratch/bobbyy/output_raw/bigbird-yahoo_answers-scorch.txt; done

# for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset imdb | tee -a /scratch/bobbyy/output_raw/bigbird-imdb-torch.txt; done
# for i in {1..15}; do python scorch_sparse_transformer.py --mode test --dataset imdb | tee -a /scratch/bobbyy/output_raw/bigbird-imdb-scorch.txt; done

# for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset ag_news | tee -a /scratch/bobbyy/output_raw/bigbird-ag_news-torch.txt; done
# for i in {1..15}; do python scorch_sparse_transformer.py --mode test --dataset ag_news | tee -a /scratch/bobbyy/output_raw/bigbird-ag_news-scorch.txt; done


for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset ag_news --gpu | tee -a /scratch/bobbyy/output_raw/bigbird-ag_news-torch_gpu.txt; done
for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset imdb --gpu | tee -a /scratch/bobbyy/output_raw/bigbird-imdb-torch_gpu.txt; done
for i in {1..15}; do python torch_sparse_transformer.py --mode test --dataset yahoo_answers --gpu | tee -a /scratch/bobbyy/output_raw/bigbird-yahoo_answers-torch_gpu.txt; done
