from typing import Optional

from numpy import dtype

import geopandas as gpd
import h3
import pandas as pd
import libpysal as lps
from esda.moran import Moran
from esda.moran import Moran_Local
from shapely.geometry import box, mapping, Point


def fill_hex_grid(
    gdf: gpd.GeoDataFrame, geom_column: str = "geometry"
) -> gpd.GeoDataFrame:
    bbox = gdf.total_bounds
    # Pandas somehow mangles Geopandas geometry column types so that the types
    # become mixed after concatenation and may cause TypeErrors, i.e. some
    # Shapely geometries may be cast as strings in the process. We have to
    # concatenate regular dataframes instead and reconstruct a geodataframe
    # from the hex indices afterwards. Utterly stupid.
    df = gdf.drop(columns=['geometry'])

    bbox_polygon = box(*bbox)
    hex_column = next(
            (col for col in df.columns if col.startswith("hex")), False
        )
    if not hex_column:
        raise AssertionError("Cannot calculate clusters, hex column not found.")
    resolution = int(hex_column.replace("hex", ""))
    # H3 polyfill needs geojson-like stuff. geo_json_conformant switches coordinate order
    hexes_in_bbox = h3.polyfill(mapping(bbox_polygon), resolution, geo_json_conformant=True)
    # Add only missing hexes here
    missing_hexes = set(hexes_in_bbox).difference(df[hex_column])
    missing_df = pd.DataFrame(
        list(missing_hexes),
        columns=[hex_column]
    ).set_index(hex_column, drop=False)
    columns_to_add = df.columns.difference(missing_df.columns)
    for column in columns_to_add:
        # Just add zeroes for missing index values
        missing_df.insert(0, column, 0)
    combined_df = pd.concat((df, missing_df))

    # Add centroid geometries and reconstruct the geodataframe
    centroid_lat_lon = [h3.h3_to_geo(hex) for hex in combined_df[hex_column]]
    centroids = [Point(geom[1], geom[0]) for geom in centroid_lat_lon]
    combined_gdf = gpd.GeoDataFrame(combined_df)
    combined_gdf = combined_gdf.set_geometry(centroids)
    return combined_gdf


def generate_clusters(
    gdf: gpd.GeoDataFrame,
    col: str, crs: Optional[int] = None,
    alpha: float = 0.005,
    geom_column: str = "geometry"
) -> gpd.GeoDataFrame:
    """Calculates spatial clusters/outliers based on a column in a geofataframe


    Workflow:

    1.  Create a spatial weights matrix
    2.  Create a spatially lagged version of the variable of interest
    3.  Calculate global spatial autocorrelation metrics
    4.  Calculate local spatial autocorrelation (the clusters) using LISA
        (local indicators of spatial autocorrelation)
    5.  Join data to original gdf

    While the code should work for any geodataframe, the current workflow is
    based on the assumption that the data being analyzed is in a hexagonal
    grid. This means we have polygons of approximately uniform weights. The
    https://pysal.org/libpysal/generated/libpysal.weights.Rook.html weighting
    calculates weights between all polygons that share an edge. Note that this
    requires the grid is filled with polygons, i.e. we don't have islands in
    the grid.


    Input:

    gdf     The source geodataframe, should be a hexagonal grid if using this
            script as is
    crs     A coordinate reference system, EPSG code
    col     The column with the data being modeled
    alpha   The threshold of statistical significance to be used when determing
            whether a cell is a cluster/outlier or not. Defaults to 0.005. Such
            a low value is used because our data typically contains large contrasts
            between areas of zero index (forest, seas) and built-up areas.
                - Larger values show the boundary between built-up and nature
                - Smaller values show contrasts within built-up areas


    The output is the original dataframe with 2 new columns:

    quadrant        The quadrant to which the observation belongs to:
                    LL = low clusters = low values surrounded by low values
                    HH = high clusters = high values surrounded by high values
                    LH = low outliers = low values surrounded by high values
                    HL = high outliers = high values surrounded by low values
    significant     Whether the quadrant information is statistically
                    significant. The significance will depend on the number of
                    iterations and the random seed used in the process, as
                    polygons at the edge of significance may get slightly
                    different values at different runs.
    """

    # Project
    if crs:
        gdf = gdf.to_crs(crs)

    # The cluster algorithm fails if there are islands in the data, i.e. we must have a full grid.
    # This means filling the bbox of the geodataframe with zero values at each missing hex.
    # Zero index value indicates none of the datasets had any values on the hex.
    # Note that
    # 1) the datasets may have different bboxes and
    # 2) they may be sparse.
    # We cannot make any assumptions of the form of the data, other than that it is aggregated
    # in hex grid.
    gdf_filled = fill_hex_grid(gdf, geom_column=geom_column)

    # Compute spatial weights and row-standardize
    weights = lps.weights.Rook.from_dataframe(gdf_filled, geom_col=geom_column)
    weights.set_transform("R")

    # Compute spatial lag
    y = gdf_filled[col]
    y_lag = lps.weights.lag_spatial(weights, y)
    col_lag = f"{col}_lag"
    data_lag = pd.DataFrame(data={col: y, col_lag: y_lag})

    # Global spatial autocorrelation
    mi = Moran(data_lag[col], weights)
    p_value = mi.p_sim
    print(
        "\nGlobal spatial autocorrelation:\n"
        + "Moran's I:   "
        + str(round(mi.I, 3))
        + "\np-value:     "
        + str(round(p_value, 3))
    )

    # Calculate LISA values
    lisa = Moran_Local(
        data_lag[col],
        weights,
        permutations=100000,
        # seed=1             # Use this if absolute repoducibility is needed
    )

    # identify whether each observation is significant or not
    data_lag["significant"] = lisa.p_sim < alpha

    # identify the quadrant each observation belongs to
    data_lag["quadrant"] = lisa.q
    data_lag["quadrant"] = data_lag["quadrant"].replace(
        {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    )

    # Print info
    print(
        "\nDistribution of clusters/outliers (quadrants):\n"
        + str(data_lag["quadrant"].sort_values().value_counts())
    )
    print(
        "\nSignificant clusters (using significance threshold "
        + str(alpha)
        + "):\n"
        + str(data_lag["significant"].value_counts())
    )

    # Merge original gdf and LISA quadrants data together
    gdf_clusters = gdf_filled.merge(
        data_lag[["quadrant", "significant"]],
        how="left",
        left_index=True,
        right_index=True
    )

    return gdf_clusters
