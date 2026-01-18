import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import logging

info = logging.info
error = logging.error


class FaceNetModel:
    def __init__(self, image_path, model_file_path):
        self.image_path = image_path
        self.model_file_path = model_file_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.class_embeddings = {}
        self._load_model()

    # ================= LOAD FACENET =================
    def _load_model(self):
        if not os.path.exists(self.model_file_path):
            raise FileNotFoundError(f"FaceNet model not found: {self.model_file_path}")

        self.model = InceptionResnetV1(pretrained=None, classify=False).to(self.device)

        state_dict = torch.load(self.model_file_path, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logits.")}

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        info("[OK] FaceNet loaded (embedding-only)")

    # ================= GET EMBEDDING =================
    def _get_emb(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            return None

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(img)
            emb = emb / emb.norm(dim=1, keepdim=True)

        return emb.cpu().numpy()[0]

    # ================= BUILD DATABASE =================
    def build_class_embeddings(self):
        self.class_embeddings = {}

        for person in os.listdir(self.image_path):
            folder = os.path.join(self.image_path, person)
            if not os.path.isdir(folder):
                continue

            embs = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.png')):
                    emb = self._get_emb(os.path.join(folder, f))
                    if emb is not None:
                        embs.append(emb)

            if embs:
                self.class_embeddings[person] = np.mean(embs, axis=0)

        info(f"[OK] Built embeddings: {list(self.class_embeddings.keys())}")

    # ================= PREDICT PERSON =================
    def predict_person(self, image_path, threshold=0.9):
        if not self.class_embeddings:
            self.build_class_embeddings()

        emb = self._get_emb(image_path)
        if emb is None:
            return "Unknown", 9999.0

        best_dist = 9999
        best_cls = "Unknown"

        for cls, cls_emb in self.class_embeddings.items():
            dist = np.linalg.norm(emb - cls_emb)
            if dist < best_dist:
                best_dist = dist
                best_cls = cls

        if best_dist > threshold:
            return "Unknown", float(best_dist)

        return best_cls, float(best_dist)
