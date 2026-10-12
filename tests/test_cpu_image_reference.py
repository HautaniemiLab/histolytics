"""Fixed bundled-data feature values guard numerical dependency changes."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from histolytics.data import fetch
from histolytics.nuc_feats.intensity import rgb_intensity_feats
from histolytics.nuc_feats.texture import textural_feats

_INPUT_HASHES = {
    "hgsc_nest.jpg": "4525ae5db61e7a385f24ec7ff3bf26e10f61a870c36e3c74b1a3cd436990ea57",
    "hgsc_nest_inst_mask.npz": "bde26874264819e60300edb0be133f72b8b4ce785fc064c25394d8102c346119",
    "hgsc_nest_type_mask.npz": "c90bcce7fe8be0be7a00fcc5cf7667f7b1f7ba21aa1de8bb37b2b138d862162a",
}
_ARRAY_HASHES = {
    "image": "88481e6f5a473008d6f88577d797e908f0e2068ae1b5abbf63c4e172e60a4807",
    "instances": "b9abbf110a9fdae6af666f1d185e8d853d9d24f999337d03d3fc624dec187858",
    "types": "ff43da8c2e4f3c77aa9a415d8db69a37c134034b7e48cb4ae9c05b3671917ca1",
}


def test_bundled_cpu_image_feature_reference():
    expected = pd.read_csv(
        Path(__file__).with_name("fixtures") / "cpu_image_features.csv",
        index_col="uid",
        float_precision="round_trip",
    )
    for name, expected_hash in _INPUT_HASHES.items():
        assert (
            hashlib.sha256((fetch.BASE_PATH / name).read_bytes()).hexdigest()
            == expected_hash
        )
    region = np.s_[686:814, 686:814]
    arrays = {
        "image": fetch.hgsc_cancer_he()[region],
        "instances": fetch.hgsc_cancer_inst_mask()[region],
        "types": fetch.hgsc_cancer_type_mask()[region],
    }
    for name, array in arrays.items():
        assert array.shape == ((128, 128, 3) if name == "image" else (128, 128))
        assert array.dtype == (np.uint8 if name == "image" else np.int32)
        assert hashlib.sha256(array.tobytes()).hexdigest() == _ARRAY_HASHES[name]
    image, labels, types = arrays.values()
    ids, counts = np.unique(labels, return_counts=True)
    assert ids.tolist() == [0, *expected.index.tolist()]
    assert counts.tolist() == [10626, *expected.pixel_count.tolist()]
    for uid in expected.index:
        assert np.unique(types[labels == uid]).tolist() == [
            expected.loc[uid, "type_id"]
        ]
    result = pd.concat(
        [
            rgb_intensity_feats(image, labels, metrics=("mean", "std"), device="cpu"),
            textural_feats(
                image,
                labels,
                metrics=("contrast", "dissimilarity"),
                distances=(1,),
                angles=(0,),
                device="cpu",
            ),
        ],
        axis=1,
    )
    values = expected.drop(columns=["pixel_count", "type_id"])
    assert result.index.tolist() == values.index.tolist()
    assert result.columns.tolist() == values.columns.tolist()
    assert all(result.dtypes == np.float64)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(
        result.to_numpy(), values.to_numpy(), rtol=1e-6, atol=1e-8
    )
