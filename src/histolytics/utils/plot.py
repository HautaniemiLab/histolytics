import cv2
import numpy as np

from histolytics.utils.mask_utils import bounding_box

__all__ = [
    "draw_thing_contours",
]


NUM_COLORS = {
    0: (255.0, 0.0, 55.0),
    1: (255.0, 0.0, 0.0),
    2: (0.0, 200.0, 100.0),
    3: (220.0, 220.0, 55.0),
    4: (0.0, 110.0, 155.0),
    5: (50.0, 50.0, 155.0),
    6: (220.0, 220.0, 55.0),
    7: (200.0, 50.0, 55.0),
    8: (155.0, 110.0, 155.0),
    9: (0.0, 0.0, 0.0),
}


def draw_thing_contours(
    inst_map: np.ndarray,
    image: np.ndarray,
    type_map: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """Find coloured contours for an instance labelled mask.

    Parameters:
        inst_map (np.ndarray):
            Instance segmentation map. Shape (H, W).
        image (np.ndarray)
            Original image. Shape (H, W, 3).
        type_map : np.ndarray, optional
            Semantic segmentation map. Shape (H, W).
        thickness : int, default=2
            Thickness of the contour lines
        classes : Dict[str, int], optional
            Classes dict e.g. {"bg":0, "cancer":1, "immune":2}
        colors : Dict[str, Tuple[float, float, float]], optional
            Color dict for the classes.
            E.g. {"cancer": (125., 100. ,122.), "immune": (56., 37, 160.)}

    Returns:
        np.ndarray:
            The contours overlaid on top of original image. Shape: (H, W, 3).
    """
    bg = np.copy(image)

    shape = inst_map.shape[:2]
    nuc_list = list(np.unique(inst_map))

    if 0 in nuc_list:
        nuc_list.remove(0)  # 0 is background

    for _, nuc_id in enumerate(nuc_list):
        inst = np.array(inst_map == nuc_id, np.uint8)

        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= shape[0] - 1 else y2

        inst_crop = inst[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours = cv2.findContours(inst_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]

        type_crop = type_map[y1:y2, x1:x2]
        type = np.unique(type_crop[inst_crop > 0])[0]
        inst_color = NUM_COLORS[type]

        cv2.drawContours(
            inst_bg_crop,
            contours,
            contourIdx=-1,
            color=inst_color,
            thickness=thickness,
        )

        bg[y1:y2, x1:x2] = inst_bg_crop

    return bg
