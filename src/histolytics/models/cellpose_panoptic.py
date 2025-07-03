from typing import Any, Dict

import torch
import torch.nn as nn
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.cellpose.cellpose_unet import CellPoseUnet

from histolytics.models._base_model import BaseModelPanoptic

__all__ = ["CellposePanoptic"]


def cellpose_panoptic(n_nuc_classes: int, n_tissue_classes: int, **kwargs) -> nn.Module:
    """Initialize Cellpose for panoptic segmentation.

    Cellpose:
    - https://www.nature.com/articles/s41592-020-01018-x

    Parameters
        n_nuc_classes (int):
            Number of nuclei type classes.
        n_tissue_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns:
        nn.Module: The initialized Cellpose+ U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("type", "tissue"),
        heads={
            "type": {"nuc_cellpose": 2, "nuc_type": n_nuc_classes},
            "tissue": {"tissue_type": n_tissue_classes},
        },
        **kwargs,
    )
    return cellpose_unet


class CellposePanoptic(BaseModelPanoptic):
    model_name = "cellpose_panoptic"

    def __init__(
        self,
        n_nuc_classes: int,
        n_tissue_classes: int,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """CellposePanoptic model for panoptic segmentation of nuclei and tissues.

        Note:
            - [Cellpose article](https://www.nature.com/articles/s41592-020-01018-x)

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
            model_kwargs (Dict[str, Any]):
                Additional keyword arguments for the model.
        """
        super().__init__()
        self.model = cellpose_panoptic(
            n_nuc_classes,
            n_tissue_classes,
            enc_name=enc_name,
            enc_pretrain=enc_pretrain,
            enc_freeze=enc_freeze,
            **model_kwargs,
        )

        self.device = device
        self.model.to(device)

    def set_inference_mode(
        self,
        mixed_precision: bool = True,
        postproc_kwargs: Dict[str, Any] = {"use_gpu": True},
    ) -> None:
        """Set to model inference mode."""
        self.model.eval()
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(
            postproc_method="cellpose",
            postproc_kwargs=postproc_kwargs,
        )
        self.inference_mode = True
