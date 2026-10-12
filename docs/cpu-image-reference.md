# CPU image-feature reference

The [fixed reference](validation/cpu-image-feature-reference.json) records current
behavior for two public feature APIs on the existing bundled HGSC tumor-nest data.
It gives future numerical upgrades a real-data comparison alongside the controlled
image-to-analysis composition test. It does not establish clinical correctness,
pretrained segmentation accuracy, GPU parity or performance.

The region is `(x=686, y=686, width=128, height=128)` in level-zero pixels. RGB input
is uint8 `(128, 128, 3)`; instance/type labels are int32 `(128, 128)`. Background is
zero. All 13 original positive instance IDs remain, including objects cut by the
crop boundary. Type IDs come directly from the paired bundled mask. Coordinates
are uncalibrated pixels, with x right and y down.

The reference includes RGB mean/std and GLCM contrast/dissimilarity at distance one
and angle zero, using `device="cpu"` and the existing public preprocessing. Output
IDs, column names and dtypes are checked, as are raw file hashes, decoded crop
hashes, type IDs and pixel counts. Floating values use the predefined texture
evaluation tolerances: `rtol=1e-6`, `atol=1e-8`.

Capture used the complete locked analysis/development profile on macOS ARM64 with
Python 3.12.13. Source revision, feature-source hashes and relevant package versions
are recorded in the JSON. The reference contains derived values from the existing
bundled example; its existing data provenance remains applicable.

Reproduce the check from an installed analysis profile:

```sh
HF_HUB_OFFLINE=1 uv run --no-sync pytest tests/test_cpu_image_reference.py -q
```

The same regression runs in the existing supported source suites and Linux
installed-analysis wheel job. A mismatch requires investigating input decoding,
preprocessing, object selection or feature calculation. Preserve the recorded
expected values and tolerances while evaluating an upgrade; intentional changes
require an independently reviewed explanation and revised scientific reference.
