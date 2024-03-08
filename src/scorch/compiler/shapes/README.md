# Scorched Burrito

This compiler implements the work in *Compilation of Shape Operators on Sparse Arrays* by Root, et. al.
It takes as input the CIN constructs provided by the Scorch compiler, and first lowers them to Control Flow 
Intermediate Representation (CFIR), and second lowers them to C++ IR (CPP). The final compilation phase mimics 
the Scorch prologue and epilogue, i.e., execution with PyTorch tensors.

Additionally, it includes a prototype JIT compiler that follows the JAX "pure" compilation 
approach. This enables on-the-fly rewrites and fusion.

Correctness is not guaranteed.