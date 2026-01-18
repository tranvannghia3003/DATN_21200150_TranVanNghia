import os
import random
import numpy as np
from PIL import Image
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        d_pos = F.pairwise_distance(a, p)
        d_neg = F.pairwise_distance(a, n)
        return torch.mean(F.relu(d_pos - d_neg + self.margin))

# Triplet Dataset (YOLO-CROPPED)
class TripletDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

        self.label_to_indices = {}
        for idx, lbl in enumerate(labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        anchor_idx = idx
        anchor_lbl = self.labels[idx]

        anchor_img = self._load(self.paths[idx])

        pos_idx = random.choice([
            i for i in self.label_to_indices[anchor_lbl] if i != anchor_idx
        ])
        positive_img = self._load(self.paths[pos_idx])

        neg_lbl = random.choice([x for x in self.unique_labels if x != anchor_lbl])
        neg_idx = random.choice(self.label_to_indices[neg_lbl])
        negative_img = self._load(self.paths[neg_idx])

        return anchor_img, positive_img, negative_img

# Training
class FaceNetModel:
    def __init__(self,
                 image_path,
                 batch_size=12,
                 lr=1e-5,
                 num_epochs=20,
                 ckpt="facenet_yolo_best.pth"):

        self.image_path = image_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt = ckpt

        # Transform for YOLO cropped images
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Load FaceNet
        self.model = InceptionResnetV1(pretrained="vggface2").to(self.device)

        # fine-tune last layers
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.last_linear.parameters():
            p.requires_grad = True
        for p in self.model.last_bn.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

        self.criterion = TripletLoss()
        self.class_embeddings = {}

    def _load_images(self):
        paths, labels = [], []
        for person in os.listdir(self.image_path):
            folder = os.path.join(self.image_path, person)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(("jpg", "jpeg", "png")):
                    paths.append(os.path.join(folder, f))
                    labels.append(person)
        return paths, labels

    def _tensor_from_path(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _get_embedding(self, tensor):
        with torch.no_grad():
            tensor = tensor.unsqueeze(0).to(self.device)
            emb = self.model(tensor)
            return emb[0].cpu().numpy()

    def build_class_embeddings(self):
        paths, labels = self._load_images()
        storage = {}

        for p, lbl in zip(paths, labels):
            emb = self._get_embedding(self._tensor_from_path(p))
            storage.setdefault(lbl, []).append(emb)

        self.class_embeddings = {k: np.mean(v, axis=0) for k, v in storage.items()}

    def train(self, val_ratio=0.2, threshold=1.0, metric="euclidean"):
        print("=== TRAINING TRIPLET — YOLO CROPPED DATASET ===")

        paths, labels = self._load_images()
        p_train, p_val, l_train, l_val = train_test_split(
            paths, labels, test_size=val_ratio, stratify=labels, random_state=42
        )

        loader = DataLoader(
            TripletDataset(p_train, l_train, self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        best = 0

        for epoch in range(self.num_epochs):
            self.model.train()

            # freeze BN
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

            total, steps = 0, 0

            for a, p, n in loader:
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)

                ea = self.model(a)
                ep = self.model(p)
                en = self.model(n)

                ea = ea / ea.norm(dim=1, keepdim=True)
                ep = ep / ep.norm(dim=1, keepdim=True)
                en = en / en.norm(dim=1, keepdim=True)

                loss = self.criterion(ea, ep, en)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total += loss.item()
                steps += 1

            print(f"[Epoch {epoch+1}] Loss={total/steps:.4f}")

            self.build_class_embeddings()
            val_acc = self.evaluate_subset(p_val, l_val, threshold, metric)
            print("Val Acc:", round(val_acc, 4))

            if val_acc > best:
                best = val_acc
                torch.save(self.model.state_dict(), self.ckpt)
                print("✓ Saved best →", self.ckpt)

    def evaluate_subset(self, p_list, l_list, threshold, metric):
        correct = 0
        for p, lbl in zip(p_list, l_list):
            emb = self._get_embedding(self._tensor_from_path(p))
            pred, _ = self.predict_emb(emb, threshold, metric)
            if pred == lbl:
                correct += 1
        return correct / len(p_list)

    def predict_emb(self, emb, threshold, metric):
        best_cls = "Unknown"

        if metric == "euclidean":
            best = 9999
            for cls, c in self.class_embeddings.items():
                d = np.linalg.norm(emb - c)
                if d < best:
                    best, best_cls = d, cls
            return (best_cls if best <= threshold else "Unknown"), best

        else:
            best = -1
            for cls, c in self.class_embeddings.items():
                sim = np.dot(emb, c) / (np.linalg.norm(emb)*np.linalg.norm(c))
                if sim > best:
                    best, best_cls = sim, cls
            return (best_cls if best >= threshold else "Unknown"), best

    def compute_threshold(self, paths, labels):
        self.build_class_embeddings()

        pos_dist, neg_dist = [], []

        for p, lbl in zip(paths, labels):
            emb = self._get_embedding(self._tensor_from_path(p))

            for cls, c_emb in self.class_embeddings.items():
                d = float(np.linalg.norm(emb - c_emb))
                if cls == lbl:
                    pos_dist.append(d)
                else:
                    neg_dist.append(d)

        print("\n======= DISTANCE STATISTICS =======")
        print("Positive:", "min=", np.min(pos_dist), "max=", np.max(pos_dist), "mean=", np.mean(pos_dist))
        print("Negative:", "min=", np.min(neg_dist), "max=", np.max(neg_dist), "mean=", np.mean(neg_dist))

        threshold = (np.mean(pos_dist) + np.mean(neg_dist)) / 2
        print("Recommended threshold =", threshold)

        return threshold, pos_dist, neg_dist


if __name__ == "__main__":
    model = FaceNetModel(
        image_path="D:/face_model/build/yolo_cropped_dataset",  
        batch_size=8,
        lr=1e-5,
        num_epochs=50,
        ckpt="facenet_yolo_best.pth"
    )

    model.train(
        val_ratio=0.2,
        threshold=0.9,
        metric="euclidean"
    )

    # Find best threshold
    paths, labels = model._load_images()
    model.compute_threshold(paths, labels)
