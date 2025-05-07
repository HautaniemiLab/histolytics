from typing import Any, Dict

import torch
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.hovernet.hovernet_unet import hovernet_panoptic

from histolytics.models._base_model import BaseModelPanoptic

__all__ = ["HoverNetPanoptic"]


class HoverNetPanoptic(BaseModelPanoptic):
    model_name = "hovernet_panoptic"

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
        """HovernetPanoptic model for panoptic segmentation of nuclei and tissues.

        Parameters:
            n_nuc_classes (int):
                Number of nuclei type classes.
            n_tissue_classes (int):
                Number of tissue type classes.
            enc_name (str, default="efficientnet_b5"):
                Name of the pytorch-image-models encoder.
            enc_pretrain (bool, default=True):
                Whether to use pretrained weights in the encoder.
            enc_freeze (bool, default=False):
                Freeze encoder weights for training.
            device (torch.device, default=torch.device("cuda")):
                Device to run the model on. Default is "cuda".
            model_kwargs (Dict[str, Any], default={}):
                Additional keyword arguments for the model.
        """
        super().__init__()
        self.model = hovernet_panoptic(
            n_nuc_classes,
            n_tissue_classes,
            enc_name=enc_name,
            enc_pretrain=enc_pretrain,
            enc_freeze=enc_freeze,
            **model_kwargs,
        )

        self.device = device
        self.model.to(device)

    def set_inference_mode(self, mixed_precision: bool = True) -> None:
        """Set to inference mode."""
        self.model.eval()
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(postproc_method="hovernet")
        self.inference_mode = True
