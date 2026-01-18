import os
import numpy as np
import uuid
import torch
from PIL import Image
from ultralytics import YOLO
from facenet_model.euclidean import FaceNetModel
from logger import info, error

# ====================== PATH SETTINGS ======================
file_location = os.path.abspath(__file__)
root_directory = os.path.dirname(file_location)

build_dir = os.path.join(root_directory, 'build')
dataset_dir = os.path.join(build_dir, 'dataset_recognize')

facenet_model_dir = os.path.join(build_dir, 'model')
facenet_model_file_path = os.path.join(facenet_model_dir, 'facenet_best.pth')


# ====================== IMAGE PROCESSOR ======================
class ImageProcessor:
    def __init__(self, yolo_model_path, person_name="UNKNOWN"):
        self.person_name = person_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLO
        try:
            self.yolo_model = YOLO(yolo_model_path).to(self.device)
        except:
            self.yolo_model = YOLO(yolo_model_path)

        # Load FaceNet
        self.facenet_model = FaceNetModel(
            image_path=dataset_dir,
            model_file_path=facenet_model_file_path
        )

        self.recognition_threshold = 0.9
        self.temp_dir = "temp_faces"
        os.makedirs(self.temp_dir, exist_ok=True)

    # ====================== VERIFY SINGLE IMAGE ======================
    def verify_images(self, image_path):
        try:
            if not self.facenet_model.class_embeddings:
                self.facenet_model.build_class_embeddings()

            person, distance = self.facenet_model.predict_person(
                image_path=image_path,
                threshold=self.recognition_threshold
            )

            return {
                "status": "success",
                "predicted_person": person,
                "distance": float(distance),
                "is_same_person": distance <= self.recognition_threshold
            }

        except Exception as e:
            error(f"verify_images error: {e}")
            return {"status": "error"}

    # ====================== PROCESS IMAGE ======================
    def process_image(self, image: Image.Image):
        info("Starting image processing...")

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)
        results = self.yolo_model(image_np)

        if not self.facenet_model.class_embeddings:
            self.facenet_model.build_class_embeddings()
            info("Re-built FaceNet class embeddings")

        detections = []

        for result in results:
            for box in result.boxes:
                if int(box.cls.cpu().item()) != 0:
                    continue

                conf = float(box.conf.item())
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = image.crop((x1, y1, x2, y2)).resize((160, 160))

                temp_path = os.path.join(self.temp_dir, f"{uuid.uuid4().hex}.jpg")
                face_crop.save(temp_path)

                person, distance = self.facenet_model.predict_person(
                    image_path=temp_path,
                    threshold=self.recognition_threshold
                )

                is_known = distance <= self.recognition_threshold
                if not is_known:
                    person = "Unknown"

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "person_name": person,
                    "distance": float(distance),
                    "is_known": bool(is_known)
                })

                status = " Known" if is_known else " Unknown"
                info(f"{status}: {person} (dist={distance:.4f})")

                os.remove(temp_path)

        return {
            "status": "success",
            "total_faces": len(detections),
            "detections": detections
        }

    # ====================== ADD NEW PERSON ======================
    def retrieve_image(self, image: Image.Image):
        info(f"Adding new user: {self.person_name}")

        label_dir = os.path.join(dataset_dir, self.person_name)
        os.makedirs(label_dir, exist_ok=True)

        results = self.yolo_model(np.array(image))

        for r in results:
            for box in r.boxes:
                if int(box.cls.cpu().item()) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_crop = image.crop((x1, y1, x2, y2)).resize((160, 160))

                    save_path = os.path.join(
                        label_dir,
                        f"{self.person_name}_{uuid.uuid4().hex}.jpg"
                    )
                    face_crop.save(save_path, quality=95)

                    info(f"Saved dataset image: {save_path}")

                    # reset embeddings
                    self.facenet_model.class_embeddings = {}
                    return {"status": "success", "message": "User added"}

        return {"status": "error", "message": "No face detected"}
