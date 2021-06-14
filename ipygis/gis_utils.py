from decimal import Decimal
from statistics import mean
from typing import TypedDict, Optional, List

import geopandas as gpd
from keplergl import KeplerGl
from pandas import DataFrame, read_sql
from shapely import wkb, geos

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


class Center(TypedDict):
    latitude: float
    longitude: float


class QueryResult:

    def __init__(self, gdf: gpd.GeoDataFrame, center: Optional[Center], name: Optional[str] = None):
        self.gdf = gdf
        self.is_empty = center is None
        self.center = center
        self.name = name

    @property
    def geom_type(self):
        if not self.is_empty:
            return list(self.gdf.head(1)[GEOM_COL])[0].geometryType()

    @staticmethod
    def create(rs: Union[ResultSet,Query], geom_column: str = 'geom', name: Optional[str] = None):
        gdf = to_gdf(rs, geom_column)
        center = None
        if len(gdf):
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            minx, miny, maxx, maxy = gdf.total_bounds
            center = {'latitude': (miny + maxy) / 2.0, 'longitude': (minx + maxx) / 2.0}
        return QueryResult(gdf, center, name)

    @staticmethod
    def get_center_of_all(query_results: List['QueryResult']) -> Optional[Center]:
        latitudes = [qr.center['latitude'] for qr in query_results if not qr.is_empty]
        longitudes = [qr.center['longitude'] for qr in query_results if not qr.is_empty]
        return {'latitude': mean(latitudes), 'longitude': mean(longitudes)} if len(latitudes) else None

    def __str__(self):
        return f'{self.gdf.head(1)}'


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


def generate_map(query_results: List[QueryResult], height: int = 500) -> KeplerGl:
    config = KEPLER_DEFAULT_CONFIG.copy()
    center = QueryResult.get_center_of_all(query_results)
    has_strings = list(filter(lambda qr: qr.geom_type in ['MultiLineString', 'LineString'], query_results))

    if center:
        config['config']['mapState'].update(**center)
    if has_strings:
        config['config']['visState']['layerBlending'] = 'additive'
    # TODO: determine zoom level

    map_1 = KeplerGl(height=height, config=config)
    for i, qr in enumerate(query_results, start=1):
        if not qr.is_empty:
            name = qr.name if qr.name is not None else f'data_{i}'
            map_1.add_data(data=qr.gdf, name=name)
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
