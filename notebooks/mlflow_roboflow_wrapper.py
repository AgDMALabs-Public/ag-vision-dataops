
import os
import glob
import cv2
import mlflow
import pandas as pd

from pathlib import Path


class RoboflowWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from inference_models import AutoModel

        packages = glob.glob(os.path.join(
            context.artifacts["weights"],
            "models-cache", "*", "*", "model_config.json"
        ))

        if len(packages) != 1:
            raise RuntimeError(
                f"Expected one model package, found {len(packages)}"
            )

        package_dir = os.path.dirname(packages[0])
        self.model = AutoModel.from_pretrained(package_dir)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        images_dict = []
        inference_results = []

        for idx, img in enumerate(model_input["image"]):
            image = cv2.imread(img)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img}")

            images_dict.append({
                "id": idx,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": Path(img).name
            })

            # Local inference using MLflow-stored weights
            result = self.model(image)[0]
            detections = result.to_supervision()

            predictions = []

            for i in range(len(detections)):
                x1, y1, x2, y2 = map(float, detections.xyxy[i])
                class_id = int(detections.class_id[i])

                prediction = {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": float(detections.confidence[i]),
                    "class_name": self.model.class_names[class_id],
                    "class_id": class_id
                }

                # Convert instance masks to Roboflow-style points
                if detections.mask is not None:
                    mask = detections.mask[i].astype("uint8")
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        contour = max(
                            contours, key=cv2.contourArea
                        ).reshape(-1, 2)

                        prediction["points"] = [
                            {"x": int(x), "y": int(y)}
                            for x, y in contour
                        ]

                predictions.append(prediction)

            inference_results.append({"predictions": predictions})

        return images_dict, inference_results


mlflow.models.set_model(RoboflowWrapper())
