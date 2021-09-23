from copy import deepcopy
from decimal import Decimal
from math import log2
from statistics import mean
from typing import Dict, TypedDict, Optional, List, Union

import geopandas as gpd
from keplergl import KeplerGl
from pandas import DataFrame, Series, read_sql, json_normalize
from shapely import wkb, geos
from shapely.geometry import Point
from h3 import geo_to_h3, h3_to_geo

geos.WKBWriter.defaults['include_srid'] = True

from sql.run import ResultSet
from sqlalchemy.orm import Query
from geoalchemy2.shape import to_shape

GEOM_COL = 'geometry'

KEPLER_DEFAULT_CONFIG = {'version': 'v1',
                         'config': {'visState': {'filters': [],
                                                 'layers': [],
                                                 'interactionConfig': {'tooltip': {'fieldsToShow': {},
                                                                                   'compareMode': False,
                                                                                   'compareType': 'absolute',
                                                                                   'enabled': True},
                                                                       'brush': {'size': 0.5, 'enabled': False},
                                                                       'geocoder': {'enabled': False},
                                                                       'coordinate': {'enabled': False}},
                                                 'layerBlending': 'normal',
                                                 'splitMaps': [],
                                                 'animationConfig': {'currentTime': None, 'speed': 1}},
                                    'mapState': {'bearing': 0,
                                                 'dragRotate': False,
                                                 'latitude': 37.75043,
                                                 'longitude': -122.34679,
                                                 'pitch': 0,
                                                 'zoom': 9,
                                                 'isSplit': False},
                                    'mapStyle': {'styleType': 'dark',
                                                 'topLayerGroups': {},
                                                 'visibleLayerGroups': {'label': True,
                                                                        'road': True,
                                                                        'border': False,
                                                                        'building': True,
                                                                        'water': True,
                                                                        'land': True,
                                                                        '3d building': False},
                                                 'threeDBuildingColor': [9.665468314072013,
                                                                         17.18305478057247,
                                                                         31.1442867897876],
                                                 'mapStyles': {}}}}

KEPLER_DEFAULT_LAYER_CONFIG = {
            "id": "data_1",
            "type": "geojson",
            "config": {
              "dataId": "data_1",
              "label": "data_1",
              "color": [
                241,
                92,
                23
              ],
              "columns": {
                "geojson": "geometry"
              },
              "isVisible": True,
              "visConfig": {
                "opacity": 0.8,
                "strokeOpacity": 0.8,
                "thickness": 0.5,
                "strokeColor": None,
                "colorRange": {
                  "name": "Global Warming",
                  "type": "sequential",
                  "category": "Uber",
                  "colors": [
                    "#5A1846",
                    "#900C3F",
                    "#C70039",
                    "#E3611C",
                    "#F1920E",
                    "#FFC300"
                  ]
                },
                "strokeColorRange": {
                  "name": "Global Warming",
                  "type": "sequential",
                  "category": "Uber",
                  "colors": [
                    "#5A1846",
                    "#900C3F",
                    "#C70039",
                    "#E3611C",
                    "#F1920E",
                    "#FFC300"
                  ]
                },
                "radius": 10,
                "sizeRange": [
                  0,
                  10
                ],
                "radiusRange": [
                  0,
                  50
                ],
                "heightRange": [
                  0,
                  500
                ],
                "elevationScale": 5,
                "stroked": False,
                "filled": True,
                "enable3d": False,
                "wireframe": False
              },
              "hidden": False,
              "textLabel": [
                {
                  "field": None,
                  "color": [
                    255,
                    255,
                    255
                  ],
                  "size": 18,
                  "offset": [
                    0,
                    0
                  ],
                  "anchor": "start",
                  "alignment": "center"
                }
              ]
            },
            "visualChannels": {
              "colorField": None,
              "colorScale": "quantile",
              "sizeField": None,
              "sizeScale": "linear",
              "strokeColorField": None,
              "strokeColorScale": "quantile",
              "heightField": None,
              "heightScale": "linear",
              "radiusField": None,
              "radiusScale": "linear"
            }
          }

KEPLER_DEFAULT_HEX_LAYER_CONFIG = {
            "id": "hex_data_1",
            "type": "hexagonId",
            "config": {
              "dataId": "data_1",
              "label": "hex7",
              "color": [
                34,
                63,
                154
              ],
              "columns": {
                "hex_id": "hex7"
              },
              "isVisible": True,
              "visConfig": {
                "opacity": 0.2,
                "colorRange": {
                  "name": "Global Warming",
                  "type": "sequential",
                  "category": "Uber",
                  "colors": [
                    "#5A1846",
                    "#900C3F",
                    "#C70039",
                    "#E3611C",
                    "#F1920E",
                    "#FFC300"
                  ]
                },
                "coverage": 1,
                "enable3d": False,
                "sizeRange": [
                  0,
                  500
                ],
                "coverageRange": [
                  0,
                  1
                ],
                "elevationScale": 5
              },
              "hidden": False,
              "textLabel": [
                {
                  "field": None,
                  "color": [
                    255,
                    255,
                    255
                  ],
                  "size": 18,
                  "offset": [
                    0,
                    0
                  ],
                  "anchor": "start",
                  "alignment": "center"
                }
              ]
            },
            "visualChannels": {
              "colorField": {
                "name": "size",
                "type": "integer"
              },
              "colorScale": "quantize",
              "sizeField": None,
              "sizeScale": "linear",
              "coverageField": None,
              "coverageScale": "linear"
            }
          }

class Center(TypedDict):
    latitude: float
    longitude: float


class QueryResult:

    def __init__(self, gdf: gpd.GeoDataFrame, center: Optional[Center], zoom: Optional[int], name: Optional[str] = None):
        self.gdf = gdf
        self.is_empty = center is None
        self.center = center
        self.zoom = zoom
        self.name = name

    @property
    def geom_type(self):
        if not self.is_empty:
            return list(self.gdf.head(1)[GEOM_COL])[0].geometryType()

    @staticmethod
    def flatten_dataframe(df: DataFrame, column: str) -> DataFrame:
        """
        Flattens dataframe to contain nested JSON column
        :param df: Dataframe to flatten
        :param column: Name of the nested column, e.g. tags.amenity
        """
        if "." in column:
            column, subcolumn = column.split(".", 1)
            # normalize JSON object to make queries inside
            if df[column].dtype == "O":
                df = df.merge(
                    json_normalize(df[column], max_level=1),
                    left_index=True,
                    right_index=True,
                )
                return QueryResult.flatten_dataframe(df, subcolumn)
            else:
                raise KeyError(
                    f"Cannot flatten dataframe by {column}.{subcolumn}, {column} dtype is not object"
                )
        return df

    @staticmethod
    def create(
            rs: Union[ResultSet, Query],
            geom_column: str = 'geom',
            name: Optional[str] = None,
            resolution: Optional[int] = 0,
            plot: str = "size",
            column: Optional[str] = None,
            group_by: Optional[str] = None
            ):
        gdf = to_gdf(rs, geom_column)
        center = None
        zoom = None
        if len(gdf):
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            center = get_center(gdf)
            zoom = get_zoom(gdf)

        if resolution:
            # Add H3 index
            hex_col = 'hex' + str(resolution)
            # H3 uses lat, lon
            gdf[hex_col] = gdf[GEOM_COL].apply(lambda geom: geo_to_h3(geom.y, geom.x, resolution),1)

            if group_by:
                # Rows may be grouped by any field in a JSON
                data_to_group = QueryResult.flatten_dataframe(gdf, group_by)
                if "." in group_by:
                    _, group_by = group_by.rsplit(".",1)
                # Join rows with the same value in group_by field
                data_to_group = data_to_group.groupby([hex_col, group_by], sort=False, as_index=False).size()
            elif column and "." in column:
                # Flatten the dataframe so we may calculate means etc. for the desired column
                data_to_group = QueryResult.flatten_dataframe(gdf, column)
                _, column = column.rsplit(".", 1)
            else:
                data_to_group = gdf

            groupby = data_to_group.groupby(hex_col, sort=False, as_index=False)
            if column:
                groupby = groupby[column]
            # plot = size (or mean, or median, or max, or min) of rows within the same hex
            # Available functions: https://pandas.pydata.org/docs/reference/groupby.html
            results = getattr(groupby, plot)()
            # Add centroid geometry just in case. Of course, H3 has lat lon in the wrong order again
            centroid_lat_lon = results[hex_col].map(lambda hex: h3_to_geo(hex))
            results[GEOM_COL] = [Point(geom[1], geom[0]) for geom in centroid_lat_lon]
            gdf = gpd.GeoDataFrame(results, geometry=GEOM_COL, crs=gdf.crs)
            gdf = gdf.set_index(hex_col)
            # retain hex index as column too, so it can be plotted
            gdf[hex_col] = gdf.index

        return QueryResult(gdf, center, zoom, name)

    @staticmethod
    def get_center_of_all(query_results: List['QueryResult']) -> Optional[Center]:
        latitudes = [qr.center['latitude'] for qr in query_results if not qr.is_empty]
        longitudes = [qr.center['longitude'] for qr in query_results if not qr.is_empty]
        return {'latitude': mean(latitudes), 'longitude': mean(longitudes)} if len(latitudes) else None

    @staticmethod
    def get_zoom_of_all(query_results: List['QueryResult']) -> Optional[int]:
        zooms = [qr.zoom for qr in query_results if not qr.is_empty]
        return min(zooms) if len(zooms) else None

    def __str__(self):
        return f'{self.gdf.head(1)}'


def get_zoom(gdf: gpd.GeoDataFrame) -> int:
    minx, miny, maxx, maxy = gdf.total_bounds
    # ballpark approximation of zoom around 60 °N and 2:1 viewport shape
    # - web mercator will have around 2:1 lon/lat ratio around 60 °N
    longer_axis = max(maxy - miny, (maxx - minx)/4)
    # - zoom 0 will fit ~180 lat degrees
    return log2(180/longer_axis)


def get_center(gdf: gpd.GeoDataFrame) -> Center:
    minx, miny, maxx, maxy = gdf.total_bounds
    return {'latitude': (miny + maxy) / 2.0, 'longitude': (minx + maxx) / 2.0}


def get_h3_centroids(hex_column: Series) -> List:
    centroid_lat_lon = hex_column.map(lambda hex: h3_to_geo(hex))
    return [Point(geom[1], geom[0]) for geom in centroid_lat_lon]


def to_gdf(rs: Union[ResultSet,Query], geom_column: str = 'geom') -> gpd.GeoDataFrame:
    """
    Converts sql ResultSet or SqlAlchemy Query to GeoDataFrame
    :param rs:
    :param geom_column:
    """
    if type(rs) is ResultSet:
        df: DataFrame = rs.DataFrame()
    elif type(rs) is Query:
        df: DataFrame = read_sql(rs.statement, rs.session.bind)
    if len(df) == 0:
        return gpd.GeoDataFrame()

    cols = list(df.columns)
    if geom_column not in cols:
        for col in cols:
            if 'geom' in col.lower():
                geom_column = col
                break

    if type(rs) is ResultSet:
        df[GEOM_COL] = df[geom_column].apply(lambda geom: wkb.loads(geom, hex=True))
    elif type(rs) is Query:
        df[GEOM_COL] = df[geom_column].apply(lambda geom: to_shape(geom))
    if geom_column != GEOM_COL:
        df = df.drop(columns=[geom_column])

    head = df.head(1)
    crs = geos.lgeos.GEOSGetSRID(list(head[GEOM_COL])[0]._geom)

    # Convert column types
    for col, dtype in dict(df.dtypes).items():
        if dtype == object:
            first_val = list(head[col])[0]
            if isinstance(first_val, Decimal):
                def to_number(s):
                    try:
                        s1 = float(s)
                        return s1
                    except (ValueError, TypeError):
                        return s

                df[col] = df[col].apply(to_number)

    gdf = gpd.GeoDataFrame(df, geometry=GEOM_COL, crs=crs)
    return gdf


def generate_sum_layer(
        query_results: List[QueryResult],
        columns: List[str],
        weights: List[int],
        ) -> QueryResult:
    sum_frame = None
    for result, column, weight in zip(query_results, columns, weights):
        gdf = result.gdf
        # Try to use existing hex column. Fail if multiple hex columns are present
        hex_column = next(
            (col for col in gdf.columns if col.startswith("hex")), False
        )
        if not hex_column:
            if weight:
                raise KeyError(
                    "Cannot sum dataframe without hex column. Please use zero weight for such frames."
                )
            continue
        elif sum_frame is not None and sum_frame.index.name != hex_column:
            raise KeyError(
                    f"Cannot sum dataframes with different hex sizes, {sum_frame.index.name} vs. {hex_column}"
                )
        values = gdf[column]
        max_value = values.max()
        if weight < 0:
            # negative weight means the values should be reversed, smaller is better
            gdf['normalized'] = -weight*(max_value-values)/max_value
        else:
            gdf['normalized'] = weight*values/max_value
        if sum_frame is None:
            # start with the index of the first gdf
            sum_frame = DataFrame(index=gdf[hex_column])
            sum_frame['normalized'] = gdf['normalized']
        else:
            # missing hex values equal zero
            sum_frame = sum_frame.combine(gdf['normalized'].to_frame(), (lambda x,y: x+y), fill_value=0, overwrite=False)
        # also save the original data in the sum frame
        sum_frame[result.name] = gdf['normalized']
    sum_frame.rename({'normalized': 'sum'}, axis=1, inplace=True)
    # put the columns in the correct order
    sum_frame = sum_frame[['sum', *[result.name for result in query_results]]]
    sum_frame[hex_column] = sum_frame.index
    sum_frame[GEOM_COL] = get_h3_centroids(sum_frame[hex_column])
    sum_frame = gpd.GeoDataFrame(sum_frame, geometry=GEOM_COL)
    return QueryResult(sum_frame, get_center(sum_frame), get_zoom(sum_frame), 'sum')


def generate_map(
        query_results: List[QueryResult],
        height: int = 500,
        config: Optional[Dict] = KEPLER_DEFAULT_CONFIG,
        column: Optional[Union[str, List[str]]] = None,
        weights: Optional[List[int]] = None,
        ) -> KeplerGl:
    """
    Returns Kepler map combining multiple query results.
    :param query_results: List of QueryResult objects
    :param height: Height of the map
    :param config: Kepler config dict to use for styling the map.
    :param column: Name of the field or JSON column in each QueryResult to plot.
        Note that if you have calculated some statistics of a JSON field in
        QueryResult, the dataframe is flattened already, and you must use only
        the name of the field inside the JSON. You may provide a list of names
        (same length as query_results) if you wish to plot different columns
        in different dataframes.
    :param weights: Optional. If present, will also calculate a weighted sum
    of any H3 datasets. Must have the same length as query_results. If a dataset
    has no hex column, it will not be included.
    """

    config = deepcopy(config)
    center = QueryResult.get_center_of_all(query_results)
    zoom = QueryResult.get_zoom_of_all(query_results)
    has_strings = list(filter(lambda qr: qr.geom_type in ['MultiLineString', 'LineString'], query_results))

    if center:
        config['config']['mapState'].update(**center)
    if zoom:
        config['config']['mapState']['zoom'] = zoom
    if has_strings:
        config['config']['visState']['layerBlending'] = 'additive'

    map_1 = KeplerGl(height=height)

    # empty the layer config, add point or hex layer, depending on presence of hex column
    config["config"]["visState"]["layers"] = []
    for i, qr in enumerate(query_results, start=1):
        if not qr.is_empty:
            name = qr.name if qr.name is not None else f'data_{i}'
            map_1.add_data(data=qr.gdf, name=name)
            hex_column = next(
                (col for col in qr.gdf.columns if col.startswith("hex")), False
            )
            if hex_column:
                # add hex layer
                hex_layer_config = deepcopy(KEPLER_DEFAULT_HEX_LAYER_CONFIG)
                hex_layer_config["id"] = f"{hex_column}_{name}"
                hex_layer_config["config"]["dataId"] = name
                hex_layer_config["config"]["label"] = f"{hex_column}_{name}"
                hex_layer_config["config"]["columns"]["hex_id"] = hex_column
                # Also column that the user wishes to plot may change
                if column:
                    # Allow plotting different columns for different queries, or same column for all
                    data_column = column if isinstance(column, str) else column[i-1]
                    hex_layer_config["visualChannels"]["colorField"]["name"] = data_column
                    if qr.gdf[data_column].dtype == "float64":
                        hex_layer_config["visualChannels"]["colorField"]["type"] = "real"
                    elif qr.gdf[data_column].dtype == "int64":
                        hex_layer_config["visualChannels"]["colorField"]["type"] = "integer"
                config["config"]["visState"]["layers"].append(hex_layer_config)
            else:
                # add point layer for raw data
                layer_config = deepcopy(KEPLER_DEFAULT_LAYER_CONFIG)
                layer_config["id"] = name
                layer_config["config"]["dataId"] = name
                layer_config["config"]["label"] = name
                config["config"]["visState"]["layers"].append(layer_config)
    # Generate a new queryresult summing the hex layers if weights are present
    if weights:
        if isinstance(column, str):
            column = [column] * len(weights)
        sum_layer = generate_sum_layer(query_results, column, weights)
        name = sum_layer.name
        map_1.add_data(data=sum_layer.gdf, name=name)
        hex_column = next(
            (col for col in sum_layer.gdf.columns if col.startswith("hex")), False
        )

        # Hide the individual layers if sum is displayed
        for layer_config in config["config"]["visState"]["layers"]:
            layer_config["config"]["isVisible"] = False
        # add hex layer
        hex_layer_config = deepcopy(KEPLER_DEFAULT_HEX_LAYER_CONFIG)
        hex_layer_config["id"] = name
        hex_layer_config["config"]["dataId"] = name
        hex_layer_config["config"]["label"] = name
        hex_layer_config["config"]["columns"]["hex_id"] = hex_column
        hex_layer_config["visualChannels"]["colorField"]["name"] = name
        hex_layer_config["visualChannels"]["colorField"]["type"] = "real"
        config["config"]["visState"]["layers"].append(hex_layer_config)

    map_1.config = config
    return map_1


def get_map(rs: Union[ResultSet,Query], geom_column: str = 'geom', name: Optional[str] = None, height: int = 500) -> KeplerGl:
    """
    Returns Kepler map for single query. For multiple queries or dataframes, use generate_map
    :param rs: Result set of the query, or SqlAlchemy Query
    :param geom_column: Name of the geometry column
    :param name: Name of the dataset
    :param height: Height of the map
    """
    qr = QueryResult.create(rs, geom_column, name)
    return generate_map([qr], height)


def get_h3_map(
        rs: Union[ResultSet,Query],
        resolution: int,
        geom_column: str = 'geom',
        plot: str = 'size',
        column: Optional[str] = None,
        name: Optional[str] = None,
        height: int = 500,
        config: Optional[Dict] = KEPLER_DEFAULT_CONFIG,
        group_by: Optional[str] = None,
        ) -> KeplerGl:
    """
    Returns Kepler H3 map for single query. For multiple queries or dataframes, use generate_map
    :param rs: Result set of the query, or SqlAlchemy Query
    :param resolution: Desired H3 grid resolution to aggregate rows with.
    :param geom_column: Name of the geometry column
    :param plot: GroupBy statistic to plot per hex. Default is "size", which counts point numbers.
        Available functions: https://pandas.pydata.org/docs/reference/groupby.html
    :param column: Name of the field or JSON column to plot GroupBy property for.
    :param name: Name of the dataset
    :param height: Height of the map
    :param config: Kepler config dict to use for styling H3 map layers.
    :param group_by: Name of the field to aggregate with, in addition to H3 hex.
    """
    qr = QueryResult.create(rs, geom_column, name, resolution=resolution, plot=plot, column=column, group_by=group_by)
    column = column.rsplit(".",1)[-1] if column else None
    return generate_map([qr], height, config=config, column=column)
