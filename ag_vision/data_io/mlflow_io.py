import mlflow.pyfunc
import supervision as sv
import mlflow
import json
from datetime import date
import cv2


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

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[pred_json["class_name"]],
                "bbox": [float(x_min), float(y_min), float(pred_json.get("width")), float(pred_json.get("height"))],
                "area": float(pred_json.get("width") * pred_json.get("height")),
                "segmentation": pred_json.get("points") if pred_json.get("points") is not None else [],
                "iscrowd": 0
            })

        annotation_id += 1

    return annotations


def annotate_image_with_inference_result(image, inference_result, parameters):
    model_type = parameters["model_type"]
    inference_result_json = json.loads(inference_result.json())

    if (model_type != "object_classification"):
        detections = sv.Detections.from_inference(inference_result)
        label_annotator = sv.LabelAnnotator()

        if (model_type == "object_detection"):
            bounding_box_annotator = sv.BoxAnnotator()

            annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)

        elif (model_type == "instance_segmentation"):
            mask_annotator = sv.MaskAnnotator()
            labels = [item["class_name"] for item in inference_result_json.get("predictions")]

            annotated_image = mask_annotator.annotate(
                scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)

        return annotated_image

    else:
        return


def format_output(parameters, category_map, images, image_annotations_json, annotated_images):
    results = []

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

    results.append({
        "coco_json_output": json.dumps(coco_output),
        "annotated_images": annotated_images
    })

    return results


def ml_flow_log_run(model, input_example):
    roboflow_pyfunc_model = model

    output_example = roboflow_pyfunc_model.predict(context=None, model_input=input_example)

    signature = mlflow.models.infer_signature(input_example, output_example)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="roboflow_model",
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
