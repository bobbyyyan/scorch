"""The caller's thread count, and the ceiling it puts on scorch's own teams.

scorch sizes each parallel team from the work, and with nothing set it may use every processor
the process can run on -- more than torch's pool on a hybrid CPU, where torch keeps to the
performance cores. Once a thread count has been set, every team scorch launches stays within it.
The rule and its one blind spot are stated where it is enforced, in ``scorch_policy.h``
("The caller's thread setting").
"""

import torch

import scorch_ops as _native  # type: ignore[import-not-found]


def set_num_threads(n: int) -> None:
    """Set the intra-op thread count, for torch and for every team scorch launches.

    ``torch.set_num_threads(n)``, and a record that a count was set. scorch sees a call to
    ``torch.set_num_threads`` that leaves torch's count different from its count at
    ``import scorch``, but not one made before that import, nor one that sets that same count
    again; after this call neither case exists. A count above the processors this process may
    use gives that many processors in scorch's own teams.

    Parameters
    ----------
    n : int
        The thread count, as ``torch.set_num_threads`` takes it.
    """
    torch.set_num_threads(n)
    _native.thread_policy_record_setting()
