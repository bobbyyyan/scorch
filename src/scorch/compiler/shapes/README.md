# Scorched Burrito

This compiler implements the work in *Compilation of Shape Operators on Sparse Arrays* by Root, et. al.
It takes as input the CIN constructs provided by the Scorch compiler, and first lowers them to Control Flow 
Intermediate Representation (CFIR), and second lowers them to C++ IR (CPP). The final compilation phase mimics 
the Scorch prologue and epilogue, i.e., generates code for execution with PyTorch tensors.

Additionally, it includes a prototype JIT compiler that follows the JAX "pure" compilation approach. 
This enables on-the-fly rewrites and fusion.

Correctness is not guaranteed.

# Debugging

In case this code is touched by other engineers/researchers, I've added logging support. Hopefully this will 
better help illustrate the compilation. This uses the Python built-in [logging](https://docs.python.org/3/howto/logging.html) 
library. To use this with `pytest`, simply append the flag: `--log-cli-level=DEBUG`. To use this with `python3`, 
use `--log=DEBUG`. For more advanced logging techniques, refer to the library documentation.