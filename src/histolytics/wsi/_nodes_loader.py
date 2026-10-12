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
) -> tn.Loader:
    """Create ordered batches using the supplied dataset/collate factory.

    Args:
        dataset: Map-style dataset indexed by the sampler.
        batch_size: Number of samples per batch.
        shuffle: Randomize sample order instead of using dataset order.
        num_workers: Process workers; zero maps batches in the calling thread.
        pin_memory: Pin collated tensors for accelerator transfers.
        drop_last: Omit the final incomplete batch.
        collate: Factory accepting the dataset and returning a batch-index callable.

    Returns:
        Torchdata loader yielding the collate callable's batch results.

    Raises:
        ValueError: If ``num_workers`` is negative.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
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
    if num_workers > 0:
        node = tn.Prefetcher(node, prefetch_factor=num_workers * 2)

    return tn.Loader(node)
