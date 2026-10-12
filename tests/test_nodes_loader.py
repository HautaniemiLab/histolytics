"""Batch iteration must terminate with zero or positive worker counts."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("drop_last", [False, True])
def test_nodes_loader_batches(workers, drop_last):
    code = textwrap.dedent(
        """
        import sys
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        from histolytics.torch_datasets.wsi_patch_dataset import WSIPatchMapAndCollate
        from histolytics.wsi._nodes_loader import NodesDataLoader

        workers = int(sys.argv[1])
        drop_last = sys.argv[2] == "True"
        dataset = TensorDataset(torch.arange(15).reshape(5, 3))
        loader = NodesDataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=workers,
            pin_memory=False, drop_last=drop_last, collate=WSIPatchMapAndCollate,
        )
        batches = list(loader)
        assert [len(batch) for batch in batches] == ([2, 2] if drop_last else [2, 2, 1])
        rows = 4 if drop_last else 5
        np.testing.assert_array_equal(
            np.concatenate(batches), np.arange(rows * 3).reshape(rows, 1, 3)
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(workers), str(drop_last)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_negative_worker_count_still_raises():
    import torch
    from torch.utils.data import TensorDataset

    from histolytics.torch_datasets.wsi_patch_dataset import WSIPatchMapAndCollate
    from histolytics.wsi._nodes_loader import NodesDataLoader

    dataset = TensorDataset(torch.arange(3).reshape(1, 3))
    with pytest.raises(ValueError):
        list(
            NodesDataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=-1,
                pin_memory=False,
                drop_last=False,
                collate=WSIPatchMapAndCollate,
            )
        )
