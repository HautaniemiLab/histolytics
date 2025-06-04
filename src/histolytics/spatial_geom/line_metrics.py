from typing import Union

from shapely.geometry import LineString, MultiLineString


def tortuosity(line: Union[LineString, MultiLineString]) -> float:
    """Compute the tortuosity of a line.

    Parameters:
        line (Union[LineString, MultiLineString]):
            Input shapely LineString or MultiLineString object.

    Returns:
        float:
            The tortuosity value of a line.
    """
    if isinstance(line, LineString):
        path_length = line.length
        euclidean_distance = line.distance(
            LineString([line.coords[0], line.coords[-1]])
        )
        return path_length / euclidean_distance if euclidean_distance > 0 else None
    elif isinstance(line, MultiLineString):
        path_length = line.length
        # Find the correct start and end points by sorting the coordinates
        coords = [list(geom.coords) for geom in line.geoms]
        coords = [item for sublist in coords for item in sublist]
        sorted_coords = sorted(coords, key=lambda x: (x[0], x[1]))
        start_point = sorted_coords[0]
        end_point = sorted_coords[-1]
        euclidean_distance = LineString([start_point, end_point]).length
        return path_length / euclidean_distance if euclidean_distance > 0 else None
    else:
        return None
