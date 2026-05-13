import mlflow.pyfunc
import supervision as sv
import mlflow
import json
from datetime import date
import cv2
import os


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
    
    output_dir = os.path.join(output_path, parameters["model_id"])
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = output_path + parameters["model_id"] + "/" + parameters["output_file_name"] + ".json"
    print("Writing output to: " + final_output_path) 
    # can't use dbutils.fs.put here do to spark limitations
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False)

    return


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
        output_dir = os.path.join(images_dir, model_id, "annotated_images")
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

