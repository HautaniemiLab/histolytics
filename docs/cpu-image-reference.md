# CPU image-feature regression

The [compact fixture](../tests/fixtures/cpu_image_features.csv) retains the original
13 nucleus IDs, pixel counts, type IDs, RGB mean/std and GLCM contrast/dissimilarity
values from the bundled HGSC tumor-nest crop at (686, 686, 128, 128). It was converted
from the previous JSON capture without recalculating or changing expected values.

[The regression](../tests/test_cpu_image_reference.py) checks file/crop SHA-256,
RGB uint8 and int32 mask shapes/dtypes, background, IDs/classes and feature columns.
Numerical comparison remains `rtol=1e-6`, `atol=1e-8`. Coordinates are uncalibrated
level-zero pixels; partial edge objects remain included. Existing bundled-data
provenance applies. Initial capture used locked macOS ARM64 Python 3.12.13.

```sh
HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/test_cpu_image_reference.py -q
```

The regression runs in existing source and installed analysis checks. Investigate
any mismatch; do not regenerate expected values or relax tolerances to pass an
upgrade. The verbose capture and its environment details remain in
[Git history](https://github.com/HautaniemiLab/histolytics/tree/5cd94960f92c0dcddd905966df366075ceb6dd53/docs/validation).
