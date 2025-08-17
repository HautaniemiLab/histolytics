from typing import Callable

import torchdata.nodes as tn
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

__all__ = ["NodesDataLoader"]


def NodesDataLoader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    collate: Callable,
):
    # Assume we're working with a map-style dataset
    assert hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")

    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    node = tn.SamplerWrapper(sampler)
    node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

    map_and_collate = collate(dataset)
    node = tn.ParallelMapper(
        node,
        map_fn=map_and_collate,
        num_workers=num_workers,
        method="process",  # Set this to "thread" for multi-threading
        in_order=True,
    )

    if pin_memory:
        node = tn.PinMemory(node)
    node = tn.Prefetcher(node, prefetch_factor=num_workers * 2)

    return tn.Loader(node)
