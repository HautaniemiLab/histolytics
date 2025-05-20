__all__ = ["PRETRAINED_MODELS", "MODEL_CLASS_DICTS"]

PRETRAINED_MODELS = {
    "hovernet_panoptic": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/hovernet-histo-hgsc-pan-v1",
            "filename": "hovernet_hgsc_v1_efficientnet_b5.safetensors",
        },
        "cin2_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/hovernet-histo-cin2-pan-v1",
            "filename": "hovernet_cin2_v1_efficientnet_b5.safetensors",
        },
    },
    "cellpose_panoptic": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cellpose-histo-hgsc-pan-v1",
            "filename": "cellpose_hgsc_v1_efficientnet_b5.safetensors",
        },
        "cin2_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cellpose-histo-cin2-pan-v1",
            "filename": "cellpose_cin2_v1_efficientnet_b5.safetensors",
        },
    },
    "cellvit_panoptic": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cellvit-histo-hgsc-pan-v1",
            "filename": "cellvit_hgsc_v1_efficientnet_b5.safetensors",
        },
        "cin2_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cellvit-histo-cin2-pan-v1",
            "filename": "cellvit_cin2_v1_efficientnet_b5.safetensors",
        },
    },
    "stardist_panoptic": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/stardist-histo-hgsc-pan-v1",
            "filename": "stardist_hgsc_v1_efficientnet_b5.safetensors",
        },
        "cin2_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/stardist-histo-cin2-pan-v1",
            "filename": "stardist_cin2_v1_efficientnet_b5.safetensors",
        },
    },
    "cppnet_panoptic": {
        "hgsc_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cppnet-histo-hgsc-pan-v1",
            "filename": "cppnet_hgsc_v1_efficientnet_b5.safetensors",
        },
        "cin2_v1_efficientnet_b5": {
            "repo_id": "histolytics-hub/cppnet-histo-cin2-pan-v1",
            "filename": "cppnet_cin2_v1_efficientnet_b5.safetensors",
        },
    },
}

MODEL_CLASS_DICTS = {
    "cin2_v1_efficientnet_b5": {
        "nuc": {
            0: "background",
            1: "neoplastic",
            2: "inflammatory",
            3: "connective",
            4: "dead",
            5: "glandular_epithelial",
            6: "squamous_epithelial",
        },
        "tissue": {
            0: "background",
            1: "stroma",
            2: "cin",
            3: "squamous_epithelium",
            4: "glandular_epithelium",
            5: "slime",
            6: "blood",
        },
    },
    "hgsc_v1_efficientnet_b5": {
        "nuc": {
            0: "background",
            1: "neoplastic",
            2: "inflammatory",
            3: "connective",
            4: "dead",
            5: "macrophage_cell",
            6: "macrophage_nuc",
        },
        "tissue": {
            0: "background",
            1: "stroma",
            2: "omental_fat",
            3: "tumor",
            4: "hemorrage",
            5: "necrosis",
            6: "serum",
        },
    },
}
