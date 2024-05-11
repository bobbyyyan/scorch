# CORA
for i in {1..55}; do python pyg_gcn.py --dataset cora | tee -a ~/output_raw/gcn-cora-pyg.txt; done
for i in {1..55}; do python scorch_gcn.py --dataset cora | tee -a ~/output_raw/gcn-cora-scorch.txt; done
for i in {1..55}; do python dgl_gcn.py --dataset cora | tee -a ~/output_raw/gcn-cora-dgl.txt; done
for i in {1..55}; do python torch_gcn.py --dataset cora --sparse | tee -a ~/output_raw/gcn-cora-torch.txt; done

# CiteSeer
for i in {1..55}; do python pyg_gcn.py --dataset citeseer | tee -a ~/output_raw/gcn-citeseer-pyg.txt; done
for i in {1..55}; do python scorch_gcn.py --dataset citeseer | tee -a ~/output_raw/gcn-citeseer-scorch.txt; done
for i in {1..55}; do python dgl_gcn.py --dataset citeseer | tee -a ~/output_raw/gcn-citeseer-dgl.txt; done
for i in {1..55}; do python torch_gcn.py --dataset citeseer --sparse | tee -a ~/output_raw/gcn-citeseer-torch.txt; done

# PubMed
for i in {1..55}; do python pyg_gcn.py --dataset pubmed | tee -a ~/output_raw/gcn-pubmed-pyg.txt; done
for i in {1..55}; do python scorch_gcn.py --dataset pubmed | tee -a ~/output_raw/gcn-pubmed-scorch.txt; done
for i in {1..55}; do python dgl_gcn.py --dataset pubmed | tee -a ~/output_raw/gcn-pubmed-dgl.txt; done
for i in {1..55}; do python torch_gcn.py --dataset pubmed --sparse | tee -a ~/output_raw/gcn-pubmed-torch.txt; done

# OGBN-Arxiv
for i in {1..55}; do python pyg_gcn.py --dataset ogbn-arxiv | tee -a ~/output_raw/gcn-ogbn_arxiv-pyg.txt; done
for i in {1..55}; do python scorch_gcn.py --dataset ogbn-arxiv | tee -a ~/output_raw/gcn-ogbn_arxiv-scorch.txt; done
for i in {1..55}; do python dgl_gcn-ogbn_arxiv.py --dataset ogbn-arxiv | tee -a ~/output_raw/gcn-ogbn_arxiv-dgl.txt; done
for i in {1..55}; do python torch_gcn.py --dataset ogbn-arxiv --sparse | tee -a ~/output_raw/gcn-ogbn_arxiv-torch.txt; done

# Reddit
# for i in {1..55}; do python pyg_gcn.py --dataset reddit --batch-size 512 | tee -a ~/output_raw/gcn-reddit-pyg.txt; done
# for i in {1..55}; do python scorch_gcn.py --dataset reddit --batch-size 512  | tee -a ~/output_raw/gcn-reddit-scorch.txt; done
# # for i in {1..55}; do python dgl_gcn.py --dataset reddit | tee -a ~/output_raw/gcn-reddit-dgl.txt; done
# for i in {1..55}; do python torch_gcn.py --dataset reddit --sparse --batch-size 512 | tee -a ~/output_raw/gcn-reddit-torch.txt; done
