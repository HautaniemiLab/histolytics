from typing import List, Union

import pandas as pd
from torch.utils.data import Dataset

__all__ = ["MapAndCollateDF", "MapAndCollateSeries"]


class MapAndCollateDF:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, batch_of_indices: List[int]) -> Union[pd.Series, pd.DataFrame]:
        results = []
        for i in batch_of_indices:
            result = self.dataset[i]
            if isinstance(result, pd.DataFrame) and not result.empty:
                results.append(result)
            elif isinstance(result, pd.Series) and len(result) > 0:
                results.append(result)

        if not results:
            return pd.DataFrame(columns=["uid"] + results[0].columns.to_list())

        return (
            pd.concat(results, copy=False)
            .reset_index(drop=False)
            .rename(columns={"index": "uid"})
        )


class MapAndCollateSeries:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, batch_of_indices: List[int]) -> Union[pd.Series, pd.DataFrame]:
        results = []
        for i in batch_of_indices:
            result = self.dataset[i]
            if isinstance(result, pd.Series) and len(result) > 0:
                results.append(result)

        if not results:
            return pd.DataFrame(columns=["uid"] + results[0].index.to_list())

        return pd.DataFrame(results)
