from typing import Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

CLIPPED_ORIENTATIONS = [0, 90, 180, 270]


def rotate_bounding_box(
    bbox: BoundingBox, angle: int, im_size: Tuple[int, int]
) -> BoundingRectangle:
    """Rotates a bounding box by a given angle around the center of an image.

    This function takes an axis-aligned bounding box and rotates its coordinates
    by a specified angle (90, 180, or 270 degrees) within the bounds of a given
    image size. The result is a `BoundingRectangle` with the new coordinates
    of the rotated box's corners.

    Args:
        bbox: The `BoundingBox` to rotate.
        angle: The angle of rotation in degrees. Must be one of 0, 90, 180, 270.
        im_size: A tuple `(width, height)` representing the size of the image
            within which the rotation occurs.

    Returns:
        A `BoundingRectangle` representing the rotated bounding box.

    Raises:
        ValueError: If the angle is not one of the supported values.
    """
    # The box is left top width height in TOPLEFT coordinates
    # Bounding rectangle start with r_0 at the bottom left whatever the
    # coordinate system. Then other corners are found rotating counterclockwise
    bbox = bbox.to_top_left_origin(im_size[1])
    left, top, width, height = bbox.l, bbox.t, bbox.width, bbox.height
    im_w, im_h = im_size
    angle = angle % 360
    if angle == 0:
        return BoundingRectangle.from_bounding_box(bbox)
    elif angle == 90:
        r_x0 = top + height
        r_y0 = im_w - left
        r_x1 = r_x0
        r_y1 = r_y0 - width
        r_x2 = r_x1 - height
        r_y2 = r_y1
        r_x3 = r_x2
        r_y3 = r_y0
    elif angle == 180:
        r_x0 = im_w - left
        r_y0 = im_h - (top + height)
        r_x1 = r_x0 - width
        r_y1 = r_y0
        r_x2 = r_x1
        r_y2 = r_y1 + height
        r_x3 = r_x0
        r_y3 = r_y2
    elif angle == 270:
        r_x0 = im_h - (top + height)
        r_y0 = left
        r_x1 = r_x0
        r_y1 = r_y0 + width
        r_x2 = r_x1 + height
        r_y2 = r_y1
        r_x3 = r_x2
        r_y3 = r_y0
    else:
        msg = (
            f"invalid orientation {angle}, expected values in:"
            f" {sorted(CLIPPED_ORIENTATIONS)}"
        )
        raise ValueError(msg)
    rectangle = BoundingRectangle(
        r_x0=r_x0,
        r_y0=r_y0,
        r_x1=r_x1,
        r_y1=r_y1,
        r_x2=r_x2,
        r_y2=r_y2,
        r_x3=r_x3,
        r_y3=r_y3,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    return rectangle
