from typing import Any, Dict

import torch
import torch.nn as nn
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.cellvit.cellvit_unet import CellVitSamUnet

from histolytics.models._base_model import BaseModelPanoptic

__all__ = ["CellVitPanoptic"]


def cellvit_panoptic(
    enc_name: str, n_nuc_classes: int, n_tissue_classes: int, **kwargs
) -> nn.Module:
    """Initialaize CellVit for panoptic segmentation.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters:
        enc_name (str):
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_nuc_classes (int):
            Number of nuclei type classes.
        n_tissue_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns:
        nn.Module: The initialized CellVitSAM+ model.
    """
    cellvit_sam = CellVitSamUnet(
        enc_name=enc_name,
        decoders=("hovernet", "type", "tissue"),
        heads={
            "hovernet": {"nuc_hovernet": 2},
            "type": {"nuc_type": n_nuc_classes},
            "tissue": {"tissue_type": n_tissue_classes},
        },
        **kwargs,
    )

    return cellvit_sam


class CellVitPanoptic(BaseModelPanoptic):
    model_name = "cellvit_panoptic"

    def __init__(
        self,
        n_nuc_classes: int,
        n_tissue_classes: int,
        enc_name: str = "samvit_base_patch16",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """CellVitPanoptic model for panoptic segmentation of nuclei and tissues.

        Note:
            [CellVit article](https://arxiv.org/abs/2306.15350)

        Parameters:
            n_nuc_classes (int):
                Number of nuclei type classes.
            n_tissue_classes (int):
                Number of tissue type classes.
            enc_name (str):
                Name of the pytorch-image-models encoder.
            enc_pretrain (bool):
                Whether to use pretrained weights in the encoder.
            enc_freeze (bool):
                Freeze encoder weights for training.
            device (torch.device):
                Device to run the model on.
            model_kwargs (dict):
                Additional keyword arguments for the model.
        """
        super().__init__()
        self.model = cellvit_panoptic(
            n_nuc_classes=n_nuc_classes,
            n_tissue_classes=n_tissue_classes,
            enc_name=enc_name,
            enc_pretrain=enc_pretrain,
            enc_freeze=enc_freeze,
            **model_kwargs,
        )

        self.device = device
        self.model.to(device)

    def set_inference_mode(self, mixed_precision: bool = True) -> None:
        """Set model to inference mode."""
        self.model.eval()
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(postproc_method="hovernet")
        self.inference_mode = True
