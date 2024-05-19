# Standard Kernels

## Data

To run the benchmarks on the standard kernels, the SuiteSparse matrix collection must be downloaded, extracted, and placed in the `~/.ssgetpy` directory. We provide a sample script `download_all_ss.py` for this process.

## Running benchmarks

To run the benchmarks and generate the plots, use the following commands:

```shell
python spmv.py
python spmm.py
python spmspm.py
python sddmm.py
```
