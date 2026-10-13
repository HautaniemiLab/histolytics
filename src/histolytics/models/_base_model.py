from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from huggingface_hub import hf_hub_download
from PIL.Image import Image

from histolytics.models import MODEL_CLASS_DICTS, PRETRAINED_MODELS
from histolytics.utils._filters import gaussian_smooth


class BaseModelPanoptic:
    def __init__(self) -> None:
        self.inference_mode = False

    def set_inference_mode(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        weights: Union[str, Path],
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> "BaseModelPanoptic":
        """Load the model from pretrained weights.

        Args:
            weights: Registered checkpoint name or local checkpoint path.
            device: Device to load and run the model on. Defaults to CUDA.
            model_kwargs: Additional arguments for the model.

        Returns:
            A model with strictly loaded weights. Registered checkpoints provide
            independent nuclei/tissue class dictionaries on the returned instance;
            local checkpoints have None mappings. Loading or customizing another
            model does not change these dictionaries.

        Raises:
            ValueError: If weights is neither a file nor a registered name.
                Checkpoint loading and model construction errors propagate.

        Examples:
            >>> model = Model.from_pretrained(<str or Path to weights>, device=torch.device("cuda"))
        """
        weights_path = Path(weights)
        if not weights_path.is_file():
            if weights_path.as_posix() in PRETRAINED_MODELS[cls.model_name].keys():
                weights_path = Path(
                    hf_hub_download(
                        repo_id=PRETRAINED_MODELS[cls.model_name][weights]["repo_id"],
                        filename=PRETRAINED_MODELS[cls.model_name][weights]["filename"],
                    )
                )

            else:
                raise ValueError(
                    "Please provide a valid path. or a pre-trained model downloaded from the"
                    f" histolytics-hub. One of {list(PRETRAINED_MODELS[cls.model_name].keys())}."
                )

        enc_name, n_nuc_classes, n_tissue_classes, state_dict = cls._get_state_dict(
            weights_path, device=device
        )

        model_inst = cls(
            n_nuc_classes=n_nuc_classes,
            n_tissue_classes=n_tissue_classes,
            enc_name=enc_name,
            enc_pretrain=False,
            enc_freeze=False,
            device=device,
            model_kwargs=model_kwargs,
        )

        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_model
            except ImportError:
                raise ImportError(
                    "Please install `safetensors` package to load .safetensors files."
                )
            load_model(model_inst.model, weights_path, device.type)
        else:
            model_inst.model.load_state_dict(state_dict, strict=True)

        try:
            model_inst.nuc_classes = MODEL_CLASS_DICTS[weights]["nuc"].copy()
            model_inst.tissue_classes = MODEL_CLASS_DICTS[weights]["tissue"].copy()
        except KeyError:
            # if the model is not in the class dict, set to None
            model_inst.nuc_classes = None
            model_inst.tissue_classes = None

        return model_inst

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray, Image],
        *,
        use_sliding_win: bool = False,
        window_size: Tuple[int, int] = None,
        stride: int = None,
        save_intermediate: bool = False,
    ) -> Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]]:
        """Predict the input image or image batch.

        Parameters:
            x (Union[torch.Tensor, np.ndarray, Image]):
                Input image (H, W, C) or input image batch (B, C, H, W).
            use_sliding_win (bool):
                Whether to use sliding window for prediction.
            window_size (Tuple[int, int]):
                The height and width of the sliding window. If `use_sliding_win` is False
                this argument is ignored.
            stride (int):
                The stride for the sliding window. If `use_sliding_win` is False this
                argument is ignored.
            save_intermediate (bool, default=False):
                Whether to save intermediate results (logits). If True, the method
                returns a tuple (final predictions, intermediate results), where the
                intermediate results are the raw model outputs before argmax.

        Returns:
            Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]]:
                Dictionary of soft outputs:

                    - "nuclei": SoftInstanceOutput (type_map, aux_map).
                    - "tissue": SoftSemanticOutput (type_map).

        Examples:
            >>> my_model.set_inference_mode()
            >>> # with sliding window if image is large
            >>> x = my_model.predict(x=image, use_sliding_win=True, window_size=(256, 256), stride=128)
            >>> # without sliding window if image is small enough
            >>> x = my_model.predict(x=image, use_sliding_win=False)
        """
        if not self.inference_mode:
            raise ValueError("Run `.set_inference_mode()` before running `predict`")

        if not use_sliding_win:
            x = self.predictor.predict(
                x=x, apply_boundary_weight=False, save_intermediate=save_intermediate
            )
        else:
            if window_size is None:
                raise ValueError(
                    "`window_size` must be provided when using sliding window."
                )
            if stride is None:
                raise ValueError("`stride` must be provided when using sliding window.")

            x = self.predictor.predict_sliding_win(
                x=x,
                window_size=window_size,
                stride=stride,
                apply_boundary_weight=True,
                save_intermediate=save_intermediate,
            )

        return x

    def post_process(
        self,
        x: dict[str, SoftSemanticOutput | SoftInstanceOutput | None],
        *,
        use_async_postproc: bool = True,
        start_method: str = "threading",
        n_jobs: int = 4,
        save_paths_nuc: list[Path | str] | None = None,
        save_paths_cyto: list[Path | str] | None = None,
        save_paths_tissue: list[Path | str] | None = None,
        coords: list[tuple[int, int, int, int]] | None = None,
        class_dict_nuc: dict[int, str] | None = None,
        class_dict_cyto: dict[int, str] | None = None,
        class_dict_tissue: dict[int, str] | None = None,
        nuc_smooth_func: Callable | None = gaussian_smooth,
        cyto_smooth_func: Callable | None = gaussian_smooth,
        tissue_smooth_func: Callable | None = None,
    ) -> dict[str, list[Any] | None]:
        """Reconstruct labels from predictor outputs and optionally save geometry.

        Args:
            x: Predictor output with "nuc", "cyto", and "tissue" keys. This model
                facade requires nuclei and tissue maps; cytoplasm may be None.
                Type maps contain class IDs with shape (B, H, W), not logits.
                Instance auxiliary maps have shape (B, C_aux, H, W), where
                C_aux and values depend on the reconstruction algorithm. Optional
                binary maps contain foreground probabilities with shape (B, 1, H, W).
                The upstream processor handles tensor detachment and CPU conversion.
            use_async_postproc: Use asynchronous processing for batches larger
                than one. Single-image batches always run serially.
            start_method: Worker backend: "threading", "fork", or "spawn".
            n_jobs: Number of workers for parallel processing.
            save_paths_nuc: One destination per image for nuclei geometry, or None
                to return reconstructed nuclei arrays.
            save_paths_cyto: Cytoplasm destinations, or None to return arrays.
            save_paths_tissue: Tissue destinations, or None to return arrays.
            coords: Per-image (x, y, width, height) in pixels. Vector output is
                translated by x and y; no physical-unit conversion is performed.
            class_dict_nuc: Nuclei class-ID to class-name mapping for vector output.
            class_dict_cyto: Cytoplasm class mapping for vector output.
            class_dict_tissue: Tissue class mapping for vector output.
            nuc_smooth_func: Geometry callback for saved nuclei polygons. None
                disables smoothing; ignored for in-memory array outputs. Async
                callbacks must be pickleable, for example module-level functions.
            cyto_smooth_func: Geometry callback for saved cytoplasm polygons.
            tissue_smooth_func: Geometry callback for saved tissue polygons.

        Returns:
            The upstream processor result, unchanged, under "nuc", "cyto", and
            "tissue". In serial/synchronous processing, instance results are lists
            of (instance labels, class labels) int32 arrays with shape (H, W), and
            tissue results are lists of int32 class-label arrays. Absent outputs
            are None; saved entries are None instead of arrays. The published
            asynchronous processor's cytoplasm return is not normalized here.

        Raises:
            ValueError: If inference mode has not been enabled. Processor and
                file-writing errors propagate to the caller.

        Examples:
            >>> my_model.set_inference_mode()
            >>> x = my_model.predict(x=image, use_sliding_win=False)
            >>> x = my_model.post_process(
            ...     x,
            ...     use_async_postproc=True,
            ...     start_method="threading",
            ...     n_jobs=4,
            ... )
        """
        if not self.inference_mode:
            raise ValueError(
                "Run `.set_inference_mode()` before running `post_process`"
            )

        # if batch size is 1, run serially
        if x["tissue"].type_map.shape[0] == 1:
            return self.post_processor.postproc_serial(
                x,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                save_paths_tissue=save_paths_tissue,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
                class_dict_tissue=class_dict_tissue,
                nuc_smooth_func=nuc_smooth_func,
                cyto_smooth_func=cyto_smooth_func,
                tissue_smooth_func=tissue_smooth_func,
            )

        if use_async_postproc:
            x = self.post_processor.postproc_parallel_async(
                x,
                start_method=start_method,
                n_jobs=n_jobs,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                save_paths_tissue=save_paths_tissue,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
                class_dict_tissue=class_dict_tissue,
                nuc_smooth_func=nuc_smooth_func,
                cyto_smooth_func=cyto_smooth_func,
                tissue_smooth_func=tissue_smooth_func,
            )
        else:
            x = self.post_processor.postproc_parallel(
                x,
                start_method=start_method,
                n_jobs=n_jobs,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                save_paths_tissue=save_paths_tissue,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
                class_dict_tissue=class_dict_tissue,
                nuc_smooth_func=nuc_smooth_func,
                cyto_smooth_func=cyto_smooth_func,
                tissue_smooth_func=tissue_smooth_func,
            )

        return x

    @staticmethod
    def _get_state_dict(
        weights_path: Union[str, Path], device: torch.device = torch.device("cuda")
    ) -> None:
        """Load the model from pretrained weights."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ValueError(f"Model weights not found at {weights_path}")
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "Please install `safetensors` package to load .safetensors files."
                )
            state_dict = load_file(weights_path, device=device.type)
        else:
            state_dict = torch.load(weights_path, map_location=device)

        # infer encoder name and number of classes from state_dict
        enc_keys = [key for key in state_dict.keys() if "encoder." in key]
        enc_name = enc_keys[0].split(".")[0] if enc_keys else None
        nuc_type_head_key = next(
            key
            for key in state_dict.keys()
            if "nuc_type_head.head" in key and "weight" in key
        )
        n_nuc_classes = state_dict[nuc_type_head_key].shape[0]
        tissue_type_head_key = next(
            key
            for key in state_dict.keys()
            if "tissue_type_head.head" in key and "weight" in key
        )
        n_tissue_classes = state_dict[tissue_type_head_key].shape[0]

        return enc_name, n_nuc_classes, n_tissue_classes, state_dict
