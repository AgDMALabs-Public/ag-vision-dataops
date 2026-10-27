import mlflow.pyfunc
import mlflow
import json
import cv2
import os
import pandas as pd
import supervision as sv
import pandas as pd


from mlflow.tracking import MlflowClient
from typing import Any, Callable, Sequence
from datetime import date
from pathlib import Path



def roboflow_to_coco(predictions_json_list, image_id, category_map, parameters):
    model_type = parameters["model_type"]
    annotations = []
    annotation_id = 0

    for pred_json in predictions_json_list:
        if model_type == "object_classification":
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[pred_json["class_name"]],
                "iscrowd": 0
            })

        elif model_type == "object_detection" or model_type == "instance_segmentation":
            x_min = pred_json.get("x") - pred_json.get("width") / 2
            y_min = pred_json.get("y") - pred_json.get("height") / 2

            flat_polygon = []
            if pred_json.get("points") is not None:
                for point in pred_json.get("points"):
                    flat_polygon.extend([float(point["x"]), float(point["y"])])

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[pred_json["class_name"]],
                "bbox": [float(x_min), float(y_min), float(pred_json.get("width")), float(pred_json.get("height"))],
                "area": float(pred_json.get("width") * pred_json.get("height")),
                "segmentation": [flat_polygon] if flat_polygon != [] else [],
                "iscrowd": 0
            })

        annotation_id += 1

    return annotations


def write_coco_output_to_catalog(parameters, category_map, images, image_annotations_json, output_path):
    info = {
        "description": parameters["description"],
        "url": parameters["model_url"],
        "version": parameters["version"],
        "year": date.today().year,
        "contributor": parameters["contributor"],
        "date_created": date.today().strftime("%Y/%m/%d")
    }

    coco_output = {
        "info": info,
        "categories": [
            {"id": v, "name": k}
            for k, v in category_map.items()
        ],
        "images": images,
        "annotations": image_annotations_json
    }

    os.makedirs(output_path, exist_ok=True)
    final_output_path = output_path + "/" + parameters["output_file_name"] + ".json"
    # print("Writing output to: " + final_output_path) 
    # can't use dbutils.fs.put here do to spark limitations
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False)

    return final_output_path


def ml_flow_log_run(model_name, model, input_example):
    roboflow_pyfunc_model = model

    output_example = roboflow_pyfunc_model.predict(context=None, model_input=input_example)

    signature = mlflow.models.infer_signature(input_example, output_example)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=model,
            signature=signature,
            input_example=input_example,
            conda_env={
                "channels": ["defaults"],
                "dependencies": [
                    "pip",
                    {"pip": ["inference", "cv2"]}
                ]
            }
        )

def save_annotated_image(image_path, output_dir, annotated_image):
    base = os.path.basename(image_path)
    stem, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{stem}_annotated{ext}")

    ok = cv2.imwrite(out_path, annotated_image)
    if not ok:
        print(f"Failed to write: {out_path}")
    return

def generate_labels(detections, classes):
    labels = []
    if detections.class_id is not None and len(detections) > 0:
        confs = detections.confidence if detections.confidence is not None else [None] * len(detections)
        for cid, conf in zip(detections.class_id, confs):
            cname = classes[cid] if (cid is not None and cid < len(classes)) else "N/A"
            labels.append(f"{cname} {conf:.2f}" if conf is not None else cname)
    return labels

def annotate_images(images_dir, coco_json_file_path, model_type, model_id):
    if (model_type != "classification"):
        output_dir = os.path.join(images_dir.replace("/images", ""), "annotated_images")
        os.makedirs(output_dir, exist_ok=True)

        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=images_dir,
            annotations_path=coco_json_file_path,
            force_masks=True if model_type == "instance_segmentation" else False
        )

        label_annotator = sv.LabelAnnotator()
        classes = dataset.classes

        if (model_type == "object_detection"):
            bounding_box_annotator = sv.BoxAnnotator()

            for path, image, detections in dataset:
                labels = generate_labels(detections, classes)

                annotated_image = bounding_box_annotator.annotate(
                    scene=image, detections=detections)
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels)
                
                save_annotated_image(path, output_dir, annotated_image)

        elif (model_type == "instance_segmentation"):
            mask_annotator = sv.MaskAnnotator()

            for path, image, detections in dataset:
                labels = generate_labels(detections, classes)

                annotated_image = mask_annotator.annotate(
                    scene=image, detections=detections)
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels)
                save_annotated_image(path, output_dir, annotated_image)
    else:
        return

def existing_version_with_tag(reg_full_model_name, tag_value, tag_key="roboflow_model_id"):
    client = MlflowClient()
    versions = None

    try:
        versions = client.search_model_versions(f"name='{reg_full_model_name}'")
    except Exception:
        return None
    
    if versions:
        for v in versions:
            full = client.get_model_version(reg_full_model_name, v.version)   # tags reliably populated here
            if full.tags.get(tag_key) == tag_value:
                return full
    return None


def existing_version_with_alias(reg_full_model_name, alias_value="champion"):
    client = MlflowClient()
    versions = None

    try:
        versions = client.search_model_versions(f"name='{reg_full_model_name}'")
    except Exception:
        return None
    
    if versions:
        for v in versions:
            full = client.get_model_version(reg_full_model_name, v.version)   # tags reliably populated here
            if alias_value in full.aliases:
                return full
    return None


def register_roboflow_model(
    reg_model_name: str,
    reg_full_model_name: str,
    roboflow_model_id: str,
    model_weights_path: str | Path,
    image_samples_path: str | Path,
    image_samples: Sequence[str],
    model: Any,
    parameters: dict[str, Any],
    client: MlflowClient,
    roboflow_to_coco: Callable[..., list[dict[str, Any]]],
    write_coco_output_to_catalog: Callable[..., Any],
    alias: str = "champion",
    wrapper_path: str | Path | None = None,
) -> str:
    """
    Register a Roboflow model in MLflow and validate inference.

    Args:
        reg_model_name: Model name structured as organ__crop__task.
        reg_full_model_name: Full MLflow registered model name.
        roboflow_model_id: Original Roboflow model identifier.
        model_weights_path: Directory containing stored model artifacts.
        image_samples_path: Directory containing validation images.
        image_samples: Filenames of images used for validation.
        model: Roboflow model object with class_names and model_variant.
        parameters: Model metadata and evaluation parameters.
        client: Initialized MLflow client.
        roboflow_to_coco: Function converting predictions to COCO.
        write_coco_output_to_catalog: Function saving COCO predictions.
        alias: Alias to assign following successful validation.
        wrapper_path: Optional path to the MLflow Python model wrapper.
            Defaults to mlflow_roboflow_wrapper.py beside this module.

    Returns:
        str: Full path to the generated validation predictions JSON file.

    Raises:
        FileNotFoundError: If the wrapper file cannot be found.
        ValueError: If no validation images are provided.
        Exception: If registration, loading, or inference fails.
    """

    # Resolve the wrapper relative to this library file.
    wrapper_path = Path(
        wrapper_path or
        Path(__file__).resolve().parent / "mlflow_roboflow_wrapper.py"
    ).resolve()

    if not wrapper_path.is_file():
        raise FileNotFoundError(
            f"MLflow wrapper not found: {wrapper_path}"
        )

    # Prepare validation images.
    sample_df = pd.DataFrame({
        "image": [
            str(Path(image_samples_path) / image_name)
            for image_name in image_samples
        ]
    })

    if sample_df.empty:
        raise ValueError("No image samples provided for validation.")

    # Register the model and log evaluation metrics.
    with mlflow.start_run(run_name=f"register-{reg_model_name}"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(wrapper_path),
            model_config={
                "roboflow_model_id": roboflow_model_id
            },
            artifacts={"weights": str(model_weights_path)},
            pip_requirements=[
                "inference==1.5.2",
                "inference-models==0.37.0",
                "onnxruntime==1.21.1",
            ],
            input_example=sample_df,
            registered_model_name=reg_full_model_name,
        )

        mlflow.log_metric(
            "map50",
            parameters["overall_map50"],
            model_id=info.model_id,
        )

    version = info.registered_model_version

    # Record the original Roboflow model identifier.
    client.set_model_version_tag(
        reg_full_model_name,
        version,
        "roboflow_model_id",
        roboflow_model_id,
    )

    # Set registered-model metadata.
    organ, crop, task = reg_model_name.split("__")

    for k, v in {
        "organ": organ,
        "crop_type": crop,
        "task": task,
        "architecture": model.model_variant,
        "source": "roboflow",
    }.items():
        client.set_registered_model_tag(
            reg_full_model_name, k, v
        )

    # Load this specific version before assigning the champion alias.
    loaded = mlflow.pyfunc.load_model(
        f"models:/{reg_full_model_name}/{version}"
    )

    # Validate inference on all provided sample images.
    images, coco_predictions = loaded.predict(sample_df)

    # Convert predictions to COCO format.
    image_annotations_json = []
    predictions_path = str(image_samples_path).replace(
        "images", "predictions"
    )

    category_map = {
        cls: i + 1
        for i, cls in enumerate(model.class_names)
    }

    for image, coco_prediction in zip(images, coco_predictions):
        image_annotations_json.extend(
            roboflow_to_coco(
                coco_prediction.get("predictions"),
                image_id=image["id"],
                category_map=category_map,
                parameters=parameters,
            )
        )

    # Save validation predictions.
    write_coco_output_to_catalog(
        parameters,
        category_map,
        images,
        image_annotations_json,
        predictions_path,
    )

    predictions_file = str(
        Path(predictions_path) / "test_coco_output.json"
    )

    # Promote the model only after successful validation.
    client.set_registered_model_alias(
        reg_full_model_name, alias, version
    )

    print(
        f"Registered model '{reg_full_model_name}' "
        f"version {version} as @{alias}"
    )
    print(f"Test predictions: {predictions_file}")

    return predictions_file


def run_registered_model_inference(
    model_retrieved: mlflow.pyfunc.PyFuncModel,
    samples_df: pd.DataFrame,
    samples_path: str,
    parameters: dict[str, Any],
    roboflow_to_coco: Callable[..., list[dict[str, Any]]],
    write_coco_output_to_catalog: Callable[..., Any],
) -> str:
    """
    Run inference using an MLflow-registered model and save
    predictions in COCO format.

    Args:
        model_retrieved: Loaded MLflow model.
        samples_df: DataFrame containing image paths.
        samples_path: Directory containing input images.
        parameters: Model and inference configuration parameters.
        roboflow_to_coco: Function converting predictions to COCO.
        write_coco_output_to_catalog: Function saving COCO predictions.

    Returns:
        str: Directory containing the generated COCO predictions.
    """

    # Perform inference using the registered model.
    images, coco_predictions = model_retrieved.predict(samples_df)

    image_annotations_json = []
    predictions_path = samples_path.replace("images", "predictions")

    # Retrieve class names from the underlying model.
    category_map = {
        cls: i + 1
        for i, cls in enumerate(
            model_retrieved.unwrap_python_model().model.class_names
        )
    }

    # Convert model predictions into COCO annotations.
    for image, coco_prediction in zip(images, coco_predictions):
        image_annotations_json.extend(
            roboflow_to_coco(
                coco_prediction.get("predictions"),
                image_id=image["id"],
                category_map=category_map,
                parameters=parameters,
            )
        )

    # Save predictions in COCO format.
    write_coco_output_to_catalog(
        parameters,
        category_map,
        images,
        image_annotations_json,
        predictions_path,
    )

    predictions_file_path = str(
        Path(predictions_path) / "test_coco_output.json"
    )

    return predictions_file_path

