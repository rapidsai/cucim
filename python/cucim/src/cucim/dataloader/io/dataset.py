
import dask
import numpy as np
from dask.array.core import new_da_object
import dask.array as da


class Dataset:
    """Universal external-data wrapper for cuCIM DataLoader

    The cuCIM `Workflow` and `DataLoader`-related APIs require all
    external data to be converted to the universal `Dataset` type.  The
    main purpose of this class is to abstract away the raw format of the
    data, and to allow our Dataloader class to reliably materialize a
    `dask_cudf.Array` collection (and/or collection-based iterator)
    on demand.
    """

    def __init__(
        self,
        npy_stack_source,
        npartitions=None,
        part_size=None,
        **kwargs,
    ):
    self.npy_stack_source = npy_stack_source

    def to_da(self, indicies):
        """Returns `dask_cudf.Array` from npy_stack_source
        """
        # da = da.from_array()
        # or
        da = da.from_npy_stack(self.npy_stack_source) # set chunks?
        return da

    def to_iter(self,indices=None):
        """Convert `Dataset` object to a `cudf.Array` iterator.

        Note that this method will use `to_da` to produce a
        `dask_cudf.Array`, and materialize a single partition for
        each iteration.

        Parameters
        ----------
        indices : list(int); default None
            A specific list of partition indices to iterate over. If
            nothing is specified, all partitions will be returned in
            order (or the shuffled order, if `shuffle=True`).
        """

        return ArrayIter(
            self.to_da(indicies=indicies),
            indices=indices,
        )


class ArrayIter:
    def __init__(self, da, indices=None):
        self.indices = indices
        self._da = da

    # def __len__(self):
    #     if self.partition_lens:
    #         # Use metadata-based partition-size information
    #         # if/when it is available.  Note that this metadata
    #         # will not be correct if rows where added or dropped
    #         # after IO (within Ops).
    #         return sum(self.partition_lens[i] for i in self.indices)
    #     if len(self.indices) < self._ddf.npartitions:
    #         return len(self._ddf.partitions[self.indices])
    #     return len(self._ddf)

    def __iter__(self):
        for i in self.indices:
            yield da[i].compute(scheduler="synchronous")