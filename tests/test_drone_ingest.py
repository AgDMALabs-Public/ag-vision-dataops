# File: tests/test_ingest.py

import pandas as pd
import pytest
from ag_vision.drone.ingest import DroneDataIngest
from open_aglabs.drone.model import DroneFlight


def test_drone_data_ingest_initialization_valid_data():
    # Arrange
    platform = "aws"
    cloud_bucket = "test_bucket"
    cloud_client = None
    flight_date = "2023-03-10"
    flight_metadata_key = "metadata_key.json"
    plot_boundary_key = "boundary_key.geojson"
    gcp_key = "gcp_key.geojson"
    orthomosaic_key = "orthomosaic_key.tif"
    dem_key = "dem_key.tif"
    plot_ingest_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    # Act
    drone_ingest = DroneDataIngest(
        platform=platform,
        cloud_bucket=cloud_bucket,
        cloud_client=cloud_client,
        flight_date=flight_date,
        plot_ingest_df=plot_ingest_df,
        plot_boundary_key=plot_boundary_key,
        gcp_key=gcp_key,
        orthomosaic_key=orthomosaic_key,
        dem_key=dem_key,
        flight_metadata_key=flight_metadata_key,
    )

    # Assert
    assert drone_ingest.platform == platform
    assert drone_ingest.cloud_bucket == cloud_bucket
    assert drone_ingest.flight_date == "2023-03-10"
    assert drone_ingest.flight_metadata_key == flight_metadata_key
    assert drone_ingest.plot_boundary_key == plot_boundary_key
    assert drone_ingest.gcp_key == gcp_key
    assert drone_ingest.orthomosaic_key == orthomosaic_key
    assert drone_ingest.dem_key == dem_key
    pd.testing.assert_frame_equal(drone_ingest.plot_ingest_df, plot_ingest_df)


def test_drone_data_ingest_initialization_no_optional_data():
    # Arrange
    platform = "local"
    cloud_bucket = "local_bucket"
    cloud_client = None
    flight_date = "2025-06-15"

    # Act
    drone_ingest = DroneDataIngest(
        platform=platform,
        cloud_bucket=cloud_bucket,
        cloud_client=cloud_client,
        flight_date=flight_date,
    )

    # Assert
    assert drone_ingest.platform == platform
    assert drone_ingest.cloud_bucket == cloud_bucket
    assert drone_ingest.flight_date == "2025-06-15"
    assert drone_ingest.flight_metadata is None
    assert drone_ingest.flight_metadata_key is None
    assert drone_ingest.plot_boundary_key is None
    assert drone_ingest.gcp_key is None
    assert drone_ingest.orthomosaic_key is None
    assert drone_ingest.dem_key is None
    assert drone_ingest.plot_ingest_df is None

