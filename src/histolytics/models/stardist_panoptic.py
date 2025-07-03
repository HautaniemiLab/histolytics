from typing import Any, Dict

import torch
import torch.nn as nn
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.stardist.stardist_unet import StarDistUnet

from histolytics.models._base_model import BaseModelPanoptic

__all__ = ["StarDistPanoptic"]


def stardist_panoptic(
    n_rays: int, n_nuc_classes: int, n_tissue_classes: int, **kwargs
) -> nn.Module:
    """Initialize Stardist model for panoptic segmentation.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        n_nuc_classes (int):
            Number of nuclei type classes.
        n_tissue_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns:
        nn.Module: The initialized Panoptic Stardist model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist", "tissue"),
        heads={
            "stardist": {
                "nuc_stardist": n_rays,
                "nuc_binary": 1,
                "nuc_type": n_nuc_classes,
            },
            "tissue": {"tissue_type": n_tissue_classes},
        },
        **kwargs,
    )

    return stardist_unet


class StarDistPanoptic(BaseModelPanoptic):
    model_name = "stardist_panoptic"

    def __init__(
        self,
        n_nuc_classes: int,
        n_tissue_classes: int,
        n_rays: int = 32,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """StardistPanoptic model for panoptic segmentation of nuclei and tissues.

        Note:
            [Stardist article](https://arxiv.org/abs/1806.03535)

        Parameters:
            n_nuc_classes (int):
                Number of nuclei type classes.
            n_tissue_classes (int):
                Number of tissue type classes.
            n_rays (int):
                Number of rays for the Stardist model.
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
        self.model = stardist_panoptic(
            n_rays=n_rays,
            n_nuc_classes=n_nuc_classes,
            n_tissue_classes=n_tissue_classes,
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
        postproc_kwargs: Dict[str, Any] = {"trim_bboxes": True},
    ) -> None:
        """Set model to inference mode."""
        self.model.eval()
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(
            postproc_method="stardist",
            postproc_kwargs=postproc_kwargs,
        )
        self.inference_mode = True
