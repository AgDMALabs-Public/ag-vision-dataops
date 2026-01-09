import ag_vision.constants.paths as pth


def test_format_date_string_valid_datetime():
    # Assert
    expected = "2023-10-07"
    actual = pth._format_date_string("2023-10-07 12:34:56")
    assert expected == actual


def test_format_datetime_string_valid_datetime():
    # Assert
    expected = "2023-10-07 12:34:56"
    actual = pth._format_datetime_string("2023-10-07 12:34:56")
    assert expected == actual


def test_generate_metadata_path_from_file_name_multiple_dots():
    data_path = "data/file.name.csv"
    metadata_path = pth.generate_metadata_path_from_file_name(data_path)
    expected = "data/file.name.json"
    assert metadata_path == expected


def test_season_code():
    expected = '2023:usa:corn:winter'
    output = pth.season_code(2023, 'usa', 'corn', 'winter')
    assert expected == output


def test_location_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c'
    output = pth.location_path(project="project_a",
                               site="arusha",
                               trial="Bean_Breading",
                               season='2023:usa:corn:winter',
                               field="field_b",
                               location="loc_c")
    assert expected == output


def test_trial_image_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/phenotying/images/bean_stand_count/2023-10-07/5426589/123456789.png'
    output = pth.trial_image_path(location_path="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c",
                                  task="phenotying",
                                  protocol="bean_stand_count",
                                  date="2023-10-07",
                                  plot_id="5426589",
                                  image_name="123456789.png")
    assert expected == output


def test_scouting_image_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/scouting/images/disease_scouting/2023-10-07/123456789.png'
    output = pth.scouting_image_path(
        location_path="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c",
        task="scouting",
        protocol="disease_scouting",
        date="2023-10-07 12:34:56",
        image_name="123456789.png")
    assert expected == output


def test_annotation_image_path():
    expected = 'project_a/annotations/object_detection/plant_stand_count/batch_1/images/123456789.png'
    output = pth.annotation_image_path(project="project_a",
                                       annotation_type="object_detection",
                                       task_name="plant_stand_count",
                                       batch_name="batch_1",
                                       f_name="123456789.png")
    assert expected == output


def test_annotation_path():
    expected = 'project_a/annotations/object_detection/plant_stand_count/batch_1/2023-10-07/123456789.json'
    output = pth.annotation_path(project="project_a",
                                 annotation_type="object_detection",
                                 task_name="plant_stand_count",
                                 batch_name="batch_1",
                                 download_date="2023-10-07",
                                 f_name="123456789.json")
    assert expected == output


def test_model_weight_path():
    expected = 'project_a/models/plant_stand_count/v1/weights/weights.h5'
    output = pth.model_weight_path(project="project_a",
                                   model_name="plant_stand_count",
                                   version="v1",
                                   f_name="weights.h5")
    assert expected == output


def test_model_dataset_dir():
    expected = 'project_a/models/plant_stand_count/v1/dataset'
    output = pth.model_dataset_dir(project="project_a",
                                   model_name="plant_stand_count",
                                   version="v1")
    assert expected == output


def test_trial_note_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/phenotying/notes/bean_stand_count/2023-10-07/5426589/123456789.txt'
    output = pth.trial_note_path(location_path="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c",
                                 task="phenotying",
                                 protocol="bean_stand_count",
                                 date="2023-10-07",
                                 plot_id="5426589",
                                 note_name="123456789.txt")

    assert expected == output


def test_scouting_note_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/disease_scouting/notes/field_disease_scouting/2023-10-07/123456789.txt'
    output = pth.scouting_note_path(location_path="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c",
                                    task="disease_scouting",
                                    protocol="field_disease_scouting",
                                    date="2023-10-07",
                                    note_name="123456789.txt")

    assert expected == output


def test_drone_mission_dir():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a'
    output = pth.drone_mission_dir(location_path="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c",
                                   mission_name="stand_count_a")
    assert expected == output


def test_drone_study_boundary_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/field_data/study_boundary.geojson'
    output = pth.drone_study_boundary_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a")


def test_drone_plot_boundary_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/field_data/plot_boundary.geojson'
    output = pth.drone_plot_boundary_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a")

    assert expected == output


def test_drone_flight_details_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/2023-10-07/flight_details.json'
    output = pth.drone_flight_details_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a",
        flight_date='2023-10-07'
    )
    assert expected == output


def test_drone_flight_ground_control_point_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/field_data/ground_control_points.geojson'
    output = pth.drone_mission_ground_control_point_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a",
    )
    assert expected == output


def test_drone_flight_raw_data_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/2023-10-07/raw_data/rgb/123456789.png'
    output = pth.drone_raw_flight_data(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a",
        flight_date='2023-10-07',
        camera='rgb',
        file_name="123456789.png")
    assert expected == output


def test_drone_flight_orthomosaic_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/2023-10-07/orthomosaic/agisoft_2023-10-07/rgb/test.tiff'
    output = pth.drone_flight_orthomosaic_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a",
        flight_date='2023-10-07',
        method="agisoft",
        ortho_date="2023-10-07",
        camera='rgb',
        image_name='test.tiff')
    assert expected == output


def test_drone_plot_image_path():
    expected = 'project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a/2023-10-07/plot_image/2023-10-07 12:34:56/rgb/5426589/f123456789.png'
    output = pth.drone_flight_plot_image_path(
        mission_dir="project_a/arusha/bean_breading/2023:usa:corn:winter/field_b/loc_c/drone/stand_count_a",
        flight_date='2023-10-07',
        datetime="2023-10-07 12:34:56",
        plot_id="5426589",
        camera='rgb',
        image_name="123456789.png")
    assert expected == output

