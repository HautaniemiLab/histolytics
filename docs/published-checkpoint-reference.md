# Published checkpoint CPU reference

This reference covers one HGSC HoVer-Net/EfficientNet-B5 checkpoint and one
128×128 crop of the existing bundled tumor-nest RGB image. It establishes loading
and output behavior for future model-stack upgrades. It does not measure clinical
accuracy, whole-slide stitching, GPU behavior or every supported model.

[Checkpoint identity and capture settings](validation/hgsc-hovernet-reference.json)
record the immutable Hugging Face revision, filename, 153,801,344-byte size and
SHA-256. The [published model card](https://huggingface.co/histolytics-hub/hovernet-histo-hgsc-pan-v1/tree/976ae141db64d800e11ff6c7d84b59d9f2527ba1)
provides architecture, class definitions and Apache-2.0 licensing. The checkpoint
is retained in a local cache; the repository records small derived output arrays.

The public `HoverNetPanoptic.from_pretrained` API strictly loaded the pinned local
file with `enc_pretrain=False` and CPU execution. Local-file loading leaves automatic
class mappings unset; this reference explicitly records the registered HGSC mapping
associated with the verified artifact. No weight tensors or loading code changed.

The crop origin is `(686, 686)` in uncalibrated level-zero pixels. Public
`MinMaxNormalization(amin=0, amax=1, copy=True)` produces float32 HWC input in [0, 1].
The predictor converts it to BCHW. Eval mode, CPU float32, two Torch threads, seed 42
and disabled mixed precision are recorded. Reconstruction uses standalone serial
HoVer-Net processing with `enhance=True`. Seven nuclei are reconstructed; these
predictions are a behavior reference, not an accuracy comparison to annotations.

[Stored outputs](validation/hgsc-hovernet-reference.npz) contain predictor nuclei
class labels/auxiliary maps, predictor tissue labels, and reconstructed nuclei
instance/class labels and tissue labels. Shapes, dtypes, class mappings and integer
arrays require exact equality. Float32 auxiliary maps use predefined upgrade
comparison tolerances `rtol=1e-4`, `atol=1e-5`. Input and artifact hashes are exact.
Investigate any mismatch before considering an intentional scientific change.

Download the filename/revision recorded in the JSON to a local cache, then run in
an explicit segmentation environment:

```sh
HF_HUB_OFFLINE=1 uv run --no-sync python tools/check_checkpoint_reference.py /path/to/hovernet_hgsc_v1_efficientnet_b5.safetensors
```

The checker reads the local file and compares against fixed outputs. Ordinary CI
tests its mismatch handling offline and keeps the existing 15 jobs. Actual artifact
capture and repeat validation were performed on locked macOS ARM64 Python 3.12.13.
Linux execution of this real artifact remains a separate validation requirement
before upgrading a Linux model environment. Expected outputs and tolerances must
not be regenerated merely to make an upgrade pass.
