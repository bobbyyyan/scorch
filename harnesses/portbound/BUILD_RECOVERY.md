# The board chains' build configuration is RECOVERABLE from the sealed objects

The peer found that 20 of 28 `chain*` ledgers record their build defines and 8 do not, and that
the 8 are almost exactly the board and recount chains. I verified it by grepping every ledger for
`SCORCH_SPMM`, `HALFVEC`, `NARROWK` and `BUILD_DEFINES`:

    records defines (20): 185 (x2), 187 (x2), 188, 191, 192 (x2), 194-204, 208
    records NOTHING  (8): 189, 190, 193, **205, 206, 207, 209, 210**

So it is a regression rather than a standing limitation, and it begins at chain205 and runs
unbroken through chain210. The peer's sharper point is the one that matters: **chain196 records
defines in four files and chain206 records none, and those two are the pair whose SuiteSparse
behind-sets overlap in 7 of 25** -- so a published comparison spans the boundary, at a width where
two candidate kernels differ by a compile-time constant.

## It is recoverable, and the recovery is from the objects, not from a re-run

The chain206/chain210 build objects are still on redwood at `/scratch/bobbyy/det1/{aa,bb,aax}/src/`
and they verify against the sealed digests exactly:

    object   sha256 (16)         .text (16)          sealed in chain210's PROVENANCE.md
    aa       7f8f78d58a8784b1    a79b5d20b322c5ac    yes, both
    bb       70c9f5473e498f50    a79b5d20b322c5ac    yes, both -- byte-identical .text, as recorded
    aax      cb0450e8d0a67c7b    9b74325277bd2fb9    chain206's relocation-only third object

So these ARE the objects the boards were measured with, bound by digests computed before transfer.

## WAS IT A HOOKS BUILD? No, and the test is positive rather than an absence

The decisive test is not "does the ledger mention hooks" but "do the hook-only `getenv` string
literals exist in the object". Every `getenv("SCORCH_NARROWK_*")`, `getenv("SCORCH_SPMM_HALFVEC")`
and `getenv("SCORCH_SPMM_ALLOC")` call sits inside `#ifdef SCORCH_TUNE_HOOKS`, so the literal is in
`.rodata` if and only if the macro was defined. Reading `aa`:

    hook-only name              occurrences
    SCORCH_SPMM_HALFVEC              0
    SCORCH_NARROWK_UNROLL            0
    SCORCH_NARROWK_EXACT_HI          0
    SCORCH_SPMM_ALLOC                0
    SCORCH_REGTILE_BASE              0
    SCORCH_MULTIROW                  0
    SCORCH_DUMMY_TESTS               0

    non-hook name, as a positive control that the search works
    SCORCH_PCORES                    1
    SCORCH_LLC_BYTES                 1
    SCORCH_SPMM_ATPARALLEL           1

Seven zeros beside three ones, so the zeros are informative and not a broken search. **chain206 and
chain210 were compiled WITHOUT `SCORCH_TUNE_HOOKS`.**

## WHAT THAT SETTLES

`spmm_halfvec` is therefore the compiled-in `SCORCH_SPMM_HALFVEC_F32`, which `scorch_policy.h:1332`
defines as 1 under an `#ifndef`, and no environment variable could have changed it because the
override does not exist in the object. So **k=4 float32 ran the half-vector `__m128` path with four
of four lanes live** on both boards, and k=2 ran the full-vector path with two of eight masked.
That is now established from the run's own artefacts rather than inferred from the shipped default,
and it confirms the occupancy candidate is refuted: the worst occupancy is k=2 at 25%, and k=2 is
not where the behind-MKL cells concentrate.

**None of the eight ledgers needs to be described as indeterminate**, provided its objects survive.
That is the useful form of the finding: the gap is in the ledgers, the information is in the
objects, and the objects are digest-bound to the ledgers.

## One number I nearly published vacuously

I first counted getenv call sites with `grep -c "call.*getenv@plt"` and got **0**, which would have
read as "no environment is consulted at all". The pattern was wrong: objdump renders the import as
a PLT stub `jmp *0x...(%rip) # d9a00 <getenv@GLIBC_2.2.5>`, and the calls target the stub's own
address rather than a symbol named `getenv@plt`. Counting calls to the stub at `0x12cf0` gives
**21**. The check that caught it was `nm -D`, which shows getenv IS imported -- one dynamic symbol
against zero matches, which cannot both be true.

The 21 is a **static** count of call sites and must not be compared with the "54 getenv per call"
figure recorded elsewhere, which is dynamic. The argument above rests on the seven absent string
literals, not on either count.
