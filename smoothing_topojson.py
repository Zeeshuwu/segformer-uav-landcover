import os
from typing import Optional

import geopandas as gpd
import topojson as tp


def smooth_polygons_topojson(
    shp_path: str,
    output_shp_path: Optional[str] = None,
    simplify_tolerance: float = 1.0,
):
    """
    Smooth polygon boundaries using TopoJSON for topology-preserving simplification.

    TopoJSON automatically ensures:
    - No overlaps between adjacent polygons
    - No gaps between adjacent polygons
    - Shared boundaries are simplified together

    Parameters
    ----------
    shp_path : str
        Input shapefile path.
    output_shp_path : Optional[str]
        Output shapefile path. If None, creates folder with "_smoothing" suffix.
    simplify_tolerance : float
        Simplification tolerance in CRS units. Higher values = more smoothing.
    """

    # Read shapefile
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Input shapefile has no features.")

    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS defined.")

    # Convert to TopoJSON topology
    # This automatically merges shared boundaries and creates a topology
    topo = tp.Topology(gdf, prequantize=False)

    # Simplify using topology-preserving algorithm
    # This ensures shared boundaries between polygons are simplified together
    # preventing gaps and overlaps
    topo_simplified = topo.toposimplify(
        simplify_tolerance,
        prevent_oversimplify=True,
        simplify_algorithm="dp",  # Douglas-Peucker algorithm
    )

    # Convert back to GeoDataFrame
    gdf_smoothed = topo_simplified.to_gdf()

    # Ensure CRS is preserved
    if gdf_smoothed.crs is None:
        gdf_smoothed = gdf_smoothed.set_crs(gdf.crs)

    # Write to shapefile
    if output_shp_path is None:
        in_dir = os.path.dirname(shp_path)
        in_basename = os.path.basename(in_dir)
        parent_dir = os.path.dirname(in_dir)
        out_parent = parent_dir if parent_dir else "."
        suffixed_dirname = f"{in_basename}_smoothing"
        out_dir = os.path.join(out_parent, suffixed_dirname)
        os.makedirs(out_dir, exist_ok=True)
        in_stem, _ = os.path.splitext(os.path.basename(shp_path))
        output_shp_path = os.path.join(out_dir, f"{in_stem}_smoothing.shp")
    else:
        # Ensure directory exists if user provided a custom output path
        out_dir = os.path.dirname(output_shp_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    gdf_smoothed.to_file(output_shp_path)


if __name__ == "__main__":
    smooth_polygons_topojson(
        shp_path="data/2025_21_sieve/21_pred_sieve.shp",
        output_shp_path=None,  # will create data/2025_21_smoothing/21_pred_smoothing.shp
        simplify_tolerance=0.5,  # simplification tolerance in CRS units
    )
