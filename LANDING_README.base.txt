nc_0820 = nc_0402 + 0038 v2 + 0043 + item 37 + item 38 (composed 2026-10-13 08:20, scorch-merge)
0038 v2 (arm4 bytes)   5db710a7131315478dd56bb20311ba80257fec1a630e4f85d19e1efb9c900e2a  /scratch/bobby/ken/work/0038-int32-position-bound-on-nc_0402.v2.patch
0043                   5a66b923b72ae071c53204aba9cb1cf1c5ff58306a1188be1bc760a8c462c322  /scratch/bobby/ken/e43/0043-build-snapshot-cache.patch
item 37 (sched fields) 304d524ca3e9660c85e04c8f771eecfa5ba136d52d9d6da12efa5c9644c9d4e5  /scratch/bobby/merge_class/sched/v7_send/tree_v7.land_v.patch
item 38 (spike sums)   2778e4254f6aed9cf65acb6ab11a12969459043d0b93875d4fb387f35931028e  /scratch/bobby/merge_class/spike/spike_sequential_sum.patch
source digest (sha256 of the sorted sha256 lines of src/ and tests/, no .so/.pyc): bd05c71a5112938520d980b268c955acb4806ace40b56a089a9f656b73a4067b
If ken0038b admits arm2 instead of arm4, exactly one line changes: the out-of-line attribute on the braced-list
validate_jit_result_shape in src/scorch/csrc/native_abi.h. It changes no Python and no emitted text.
Not built: build in place in the first job that uses this tree.
