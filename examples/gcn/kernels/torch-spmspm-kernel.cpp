template<typename int*, typename float*>
void _csr_matmult(
    const int n_row,
    const int n_col,
    const int* A1_pos,
    const int* A1_crd,
    const float* A_val,
    const int* B1_pos,
    const int* B1_crd,
    const float* B_val,

    typename int C1_pos[],
    typename int C1_crd[],
    typename float C_val[]) {
  /*
    Compute CSR entries for matrix C = A@B.

    The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
    should be compatible.

    Inputs:
      `n_row`             - number of row in A
      `n_col`             - number of columns in B
      `A1_pos[n_row+1]`   - row pointer
      `A1_crd[nnz(A)]`    - column indices
      `A_val[nnz(A)]      - nonzeros
      `B1_pos[?]`         - row pointer
      `B1_crd[nnz(B)]`    - column indices
      `B_val[nnz(B)]`     - nonzeros
    Outputs:
      `C1_pos[n_row+1]`   - row pointer
      `C1_crd[nnz(C)]`    - column indices
      `C_val[nnz(C)]`     - nonzeros

    Note:
      Output arrays C1_pos, C1_crd, and C_val must be preallocated
  */

  std::vector<int> next(n_col, -1);
  std::vector<float> sums(n_col, 0);

  int nnz = 0;

  C1_pos[0] = 0;

  for (const auto i : c10::irange(n_row)) {
    int head = -2;
    int length = 0;

    int pA1_start = A1_pos[i];
    int pA1_end = A1_pos[i + 1];

    for (const auto pA1 : c10::irange(pA1_start, pA1_end)) {
      int j = A1_crd[pA1];
      float v = A_val[pA1];

      int pB1_start = B1_pos[j];
      int pB1_end = B1_pos[j + 1];

      for (const auto pB1 : c10::irange(pB1_start, pB1_end)) {
        int k = B1_crd[pB1];

        sums[k] += v * B_val[pB1];

        if (next[k] == -1) {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }

    for (const auto jj : c10::irange(length)) {

      // NOTE: the linked list that encodes col indices
      // is not guaranteed to be sorted.
      C1_crd[nnz] = head;
      C_val[nnz] = sums[head];
      nnz++;

      int temp = head;
      head = next[head];

      next[temp] = -1; // clear arrays
      sums[temp] = 0;
    }

    // Make sure that col indices are sorted.
    // NOTE: C_val arrays are expected to be contiguous!
    auto col_indices_accessor = StridedRandomAccessor<int>(C1_crd + nnz - length, 1);
    auto val_accessor = StridedRandomAccessor<float>(C_val + nnz - length, 1);
    auto kv_accessor = CompositeRandomAccessorCPU<
      decltype(col_indices_accessor), decltype(val_accessor)
    >(col_indices_accessor, val_accessor);
    std::sort(kv_accessor, kv_accessor + length, [](const auto& lhs, const auto& rhs) -> bool {
        return get<0>(lhs) < get<0>(rhs);
    });

    C1_pos[i + 1] = nnz;
  }
}
