import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .axis import axis_angle, axis_len

__all__ = [
    "major_axis_len",
    "minor_axis_len",
    "major_axis_angle",
    "minor_axis_angle",
    "compactness",
    "circularity",
    "convexity",
    "solidity",
    "elongation",
    "eccentricity",
    "fractal_dimension",
    "shape_index",
    "rectangularity",
    "squareness",
    "equivalent_rectangular_index",
]


def major_axis_len(geom: Polygon) -> float:
    """Compute the major axis length of a geometry.

    Note:
        The major axis is the (x,y) endpoints of the longest line that
        can be drawn through the object. Major axis length is the pixel
        distance between the major-axis endpoints
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    Parameters:
        geom (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The length of the major axis.
    """
    return axis_len(geom, "major")


def minor_axis_len(geom: BaseGeometry) -> float:
    """Compute the minor axis length of a geometry.

    Note:
        The minor axis is the (x,y) endpoints of the longest line that
        can be drawn through the object whilst remaining perpendicular
        with the major-axis. Minor axis length is the pixel distance
        between the minor-axis endpoints
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    Parameters:
        geom (Basegeometry):
            Input shapely geometry object.

    Returns:
        float:
            The length of the minor axis.
    """
    return axis_len(geom, "minor")


def major_axis_angle(geom: BaseGeometry) -> float:
    """Compute the major axis angle of a geometry.

    Note:
        The major axis is the (x,y) endpoints of the longest line that
        can be drawn through the object. Major axis angle is the angle of
        the major axis with respect to the x-axis.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    Parameters:
        geom (BaseGeometry):
            Input shapely geometry object.

    Returns:
        float:
            The angle of the major axis in degrees.
    """
    return axis_angle(geom, "major")


def minor_axis_angle(geom: BaseGeometry) -> float:
    """Compute the minor axis angle of a geometry.

    Note:
        The minor axis is the (x,y) endpoints of the longest line that
        can be drawn through the object whilst remaining perpendicular
        with the major-axis. Minor axis angle is the angle of the minor
        axis with respect to the x-axis.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    Parameters:
        geom (BaseGeometry):
            Input shapely geometry object.

    Returns:
        float:
            The angle of the minor axis in **degrees**.
    """
    return axis_angle(geom, "minor")


def compactness(polygon: Polygon, **kwargs) -> float:
    """Compute the compactness of a polygon.

    Note:
        Compactness is defined as the ratio of the area of an object
        to the area of a circle with the same perimeter. A circle is the
        most compact shape. Objects that are elliptical or have complicated,
        irregular (not smooth) boundaries have larger compactness.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Compactness:**
    $$
    \\frac{4\\pi A_{poly}}{P_{poly}^2}
    $$

    where $A_{poly}$ is the area of the polygon and $P_{poly}$ is the perimeter of
    the polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The compactness value of a polygon between 0-1.
    """
    perimeter = polygon.length
    area = polygon.area

    compactness = (4 * np.pi * area) / perimeter**2

    return compactness


def circularity(polygon: Polygon, **kwargs) -> float:
    """Compute the circularity of a polygon.

    Note:
        Circularity (sometimes roundness) is the ratio of the area of
        an object to the area of a circle with the same convex perimeter.
        Circularity equals 1 for a circular object and less than 1 for
        non-circular objects. Note that circularity is insensitive to
        irregular boundaries.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Circularity:**
    $$
    \\frac{4 \\times \\pi A_{poly}}{P_{convex}^2}
    $$

    where $A_{poly}$ is the area of the polygon and $P_{convex}$ is the perimeter of
    the convex hull.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The circularity value of a polygon between 0-1.
    """
    convex_perimeter = polygon.convex_hull.length
    area = polygon.area

    circularity = (4 * np.pi * area) / convex_perimeter**2

    return circularity


def convexity(polygon: Polygon, **kwargs) -> float:
    """Compute the convexity of a polygon.

    Note:
        Convexity is the relative amount that an object differs from a
        convex object. Convexity is defined by computing the ratio of
        the perimeter of an object's convex hull to the perimeter of
        the object itself. This will take the value of 1 for a convex
        object, and will be less than 1 if the object is not convex, such
        as one having an irregular boundary.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Convexity:**
    $$
    \\frac{P_{convex}}{P_{poly}}
    $$

    where $P_{convex}$ is the perimeter of the convex hull and $P_{poly}$ is the
    perimeter of the polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The convexity value of a polygon between 0-1.
    """
    convex_perimeter = polygon.convex_hull.length
    perimeter = polygon.length

    convexity = convex_perimeter / perimeter

    return convexity


def solidity(polygon: Polygon, **kwargs) -> float:
    """Compute the solidity of a polygon.

    Note:
        Solidity measures the density of an object. It is defined as the
        ratio of the area of an object to the area of a convex hull of the
        object. A value of 1 signifies a solid object, and a value less than
        1 will signify an object having an irregular boundary, or containing
        holes.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Solidity:**
    $$
    \\frac{A_{poly}}{A_{convex}}
    $$

    where $A_{poly}$ is the area of the polygon and $A_{convex}$ is the area of the
    convex hull.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The solidity value of a polygon between 0-1.
    """
    convex_area = polygon.convex_hull.area
    area = polygon.area

    return area / convex_area


def elongation(polygon: Polygon, **kwargs) -> float:
    """Compute the elongation of a polygon.

    Note:
        Elongation is the ratio between the length and width of the
        object bounding box. If the ratio is equal to 1, the object
        is roughly square or circularly shaped. As the ratio decreases
        from 1, the object becomes more elongated.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Elongation:**
    $$
    \\frac{\\text{bbox width}}{\\text{bbox height}}
    $$

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The elongation value of a polygon between 0-1.
    """
    minx, miny, maxx, maxy = polygon.bounds

    width = maxx - minx
    height = maxy - miny

    if width <= height:
        elongation = width / height
    else:
        elongation = height / width

    return elongation


def eccentricity(polygon: Polygon, **kwargs) -> float:
    """Compute the eccentricity of a polygon.

    Note:
        Eccentricity (sometimes ellipticity) measures how far the object is
        from an ellipse. It is defined as the ratio of the length of the minor
        axis to the length of the major axis of an object. The closer the
        object is to an ellipse, the closer the eccentricity is to 1
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)

    **Eccentricity:**
    $$
    \\sqrt{1 - \\frac{\\text{minor axis}^2}{\\text{major axis}^2}}
    $$

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The eccentricity value of a polygon between 0-1.
    """
    major_ax = axis_len(polygon, "major")
    minor_ax = axis_len(polygon, "minor")
    eccentricity = np.sqrt(1 - (minor_ax**2 / major_ax**2))
    return eccentricity


def fractal_dimension(polygon: Polygon, **kwargs) -> float:
    """Compute the fractal dimension of a polygon.

    Note:
        The fractal dimension is the rate at which the perimeter of an
        object increases as the measurement scale is reduced. The fractal
        dimension produces a single numeric value that summarizes the
        irregularity of "roughness" of the feature boundary.
        - [Wirth](http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf)


    **Fractal dimension:**
    $$
    2 \\times \\frac{\\log(\\frac{P_{poly}}{4})}{\\log(A_{poly})}
    $$

    where $P_{poly}$ is the perimeter of the polygon and $A_{poly}$ is the area of the
    polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The fractal dimension value of a polygon.
    """
    perimeter = polygon.length
    area = polygon.area

    return (2 * np.log(perimeter / 4)) / np.log(area)


def shape_index(polygon: Polygon, **kwargs) -> float:
    """Compute the shape index of a polygon.

    Note:
        Basically, the inverse of circularity.

    **Shape Index:**
    $$
    \\frac{\\sqrt{\\frac{A_{poly}}{\\pi}}}{\\text{MBR}}
    $$

    where $A_{poly}$ is the area of the polygon and $\\text{MBR}$ is the radius of the
    minimum bounding radius.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The shape index value of a polygon.
    """
    r = shapely.minimum_bounding_radius(polygon)
    area = polygon.area

    return np.sqrt(area / np.pi) / r


def squareness(polygon: Polygon, **kwargs) -> float:
    """Compute the squareness of a polygon.

    Note:
        Squareness is a measure of how close an object is to a square.

    **Squareness:**
    $$
    \\left(\\frac{4*\\sqrt{A_{poly}}}{P_{poly}}\\right)^2
    $$

    where $A_{poly}$ is the area of the polygon and $P_{poly}$ is the perimeter of
    the polygon.

    Note:
        For irregular shapes, squareness is close to zero and for circular shapes close
        to 1.3. For squares, equals 1

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The squareness value of a polygon.
    """
    area = polygon.area
    perimeter = polygon.length

    return ((np.sqrt(area) * 4) / perimeter) ** 2


def rectangularity(polygon: Polygon, **kwargs) -> float:
    """Compute the rectangularity of a polygon.

    Note:
        Rectangularity is the ratio of the object to the area of the
        minimum bounding rectangle. Rectangularity has a value of 1
        for perfectly rectangular object.

    **Rectangularity:**
    $$
    \\frac{A_{poly}}{A_{MRR}}
    $$

    where $A_{poly}$ is the area of the polygon and $A_{MRR}$ is the area of the
    minimum rotated rectangle.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The rectangularity value of a polygon between 0-1.
    """
    mrr = polygon.minimum_rotated_rectangle

    return polygon.area / mrr.area


def equivalent_rectangular_index(polygon: Polygon) -> float:
    """Compute the equivalent rectangular index.

    Note:
        Equivalent rectangluar index is the deviation of a polygon from
        an equivalent rectangle.

    **ERI:**
    $$
    \\frac{\\sqrt{A_{poly}}}{A_{MRR}}
    \\times
    \\frac{P_{MRR}}{P_{poly}}
    $$

    where $A_{poly}$ is the area of the polygon, $A_{MRR}$ is the area of the
    minimum rotated rectangle, $P_{MRR}$ is the perimeter of the minimum rotated
    rectangle and $P_{poly}$ is the perimeter of the polygon.

    Parameters:
        polygon (Polygon):
            Input shapely polygon object.

    Returns:
        float:
            The ERI value of a polygon between 0-1.
    """
    mrr = polygon.minimum_rotated_rectangle

    return np.sqrt(polygon.area / mrr.area) / (mrr.length / polygon.length)
