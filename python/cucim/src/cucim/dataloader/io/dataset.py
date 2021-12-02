import torch
import queue
import threading

import dask.array as da
import cupy
import numpy as np
import time
import math
import os


class BatchQueue:
    def __init__(self, darr, batch_size):
        self.darr = darr
        self.batch_size = batch_size
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.worker = threading.Thread(
            target=self.reader,
            daemon=True,
        )      

    def __len__(self):
        # Number of batches in self.darr
        return math.ceil(self.darr.shape[0] / self.batch_size)   

    def start_worker(self):
        self.worker.start()

    def put(self, batch_index):
        # Add a new batch index to input_queue
        start_index = batch_index * self.batch_size
        self.input_queue.put(start_index)

    def get(self):
        # Get a pytorch tensor from output_queue
        return self.output_queue.get()

    def reader(self):
        # Function to be executed by the IO worker
        while True:
            start_index = self.input_queue.get()
            self.output_queue.put(
                torch.as_tensor(
                    self.darr[
                        start_index:start_index + self.batch_size
                    ].compute(scheduler="synchronous")
                )
            )

        
class IterableDaskDataset(torch.utils.data.IterableDataset):
    def __init__(self, darr, batch_size=32, prefetch=True):
        super(IterableDaskDataset).__init__()
        self.darr = darr
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.batch_queue = BatchQueue(self.darr, batch_size=batch_size)

    def __len__(self):
        # Total legth of 0th index
        return self.darr.shape[0]

    def __iter__(self):
        # Simple iteration over batches
        if self.prefetch:
            # Let BatchQueue do all the work
            nbatches = len(self.batch_queue)
            if nbatches:
                self.batch_queue.start_worker()
                self.batch_queue.put(0)
            for batch_id in range(0, nbatches):
                if batch_id < nbatches - 1:
                    # Pre-fetch the next batch
                    self.batch_queue.put(batch_id + 1)
                yield self.batch_queue.get()    
        else:
            # Iterate over batches directly
            for start in range(0, len(self), self.batch_size):
                yield torch.as_tensor(
                    self.darr[start:start+self.batch_size].compute(scheduler="synchronous")
                )    



class MapDaskDataset(torch.utils.data.Dataset):
    def __init__(self, darr):
        super(MapDaskDataset).__init__()
        self.darr = darr
    
    def __len__(self):
        return len(self.darr)

    def __getitem__(self, idx):
        return self.darr[idx].compute(scheduler="synchronous")