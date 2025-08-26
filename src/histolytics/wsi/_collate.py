from typing import List, Union

import pandas as pd
from torch.utils.data import Dataset

__all__ = ["MapAndCollate"]


class MapAndCollate:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, batch_of_indices: List[int]) -> Union[pd.Series, pd.DataFrame]:
        results = []
        for i in batch_of_indices:
            result = self.dataset[i]
            if isinstance(result, pd.DataFrame) and not result.empty:
                results.append((i, result))
            elif isinstance(result, pd.Series) and len(result) > 0:
                results.append((i, result))

        if not results:
            return []

        return results
