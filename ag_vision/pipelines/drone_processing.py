import numpy as np
import geopandas as gpd

from ag_vision.drone import utils as du
from ag_vision.constants import paths as pth
from ag_vision.data_io import databricks_io as dio
from uuid import uuid4


def crop_image(plot_geometry, img_transform, img_ortho, img_x, img_y):
    dem_mask = du.generate_mask_from_polygon(polygon=plot_geometry,
                                             img_transform=img_transform,
                                             img_x=img_x,
                                             img_y=img_y)

    masked_img = du.apply_mask_to_image(image=img_ortho,
                                        mask=dem_mask,
                                        fill_value=np.nan)

    return du.crop_image_by_mask(image=masked_img,
                                 mask=dem_mask,
                                 channel_first=True)


def crop_img_by_plot_boundary_db(plot_gdf: gpd.GeoDataFrame, img_transform, img, flight_date: str, start_datetime: str,
                                 img_type: str):
    # Make sure the correct cols are in the file.
    required_cols = ['id', 'geometry']
    for col in required_cols:
        assert col in plot_gdf.columns, f"{col} is not a column in the plot_gdf. Requiured columns are {required_cols}"

    img_x = img.shape[1]
    img_y = img.shape[2]

    for idx, row in plot_gdf.iterrows():
        cropped_dem = crop_image(plot_geometry=row['geometry'],
                                 img_transform=img_transform,
                                 img_ortho=img,
                                 img_x=img_x,
                                 img_y=img_y)

        save_path = pth.drone_flight_plot_image_path(mission_dir=str(row['mission_dir']),
                                                     flight_date=flight_date,
                                                     datetime=start_datetime,
                                                     plot_id=str(row['id']),
                                                     camera=img_type,
                                                     image_name= str(uuid4()) + '.tif')

        dio.save_tif_to_databricks(img=cropped_dem,
                                   file_name=save_path)
