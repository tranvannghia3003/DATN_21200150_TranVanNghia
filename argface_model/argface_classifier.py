import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from .argface_extract_features import FeatureExtractor
from .argface_model import ArcFaceModel
from .argface_train import ArcFaceTrainer
from logger import info


class ArcFaceClassifier:
    def __init__(self, data_path, arcface_model_dir, model_save_path):
        self.model_loaded = False
        self.data_path = data_path
        self.arcface_model_dir = arcface_model_dir
        self.model_save_path = model_save_path

        self.feature_extractor = FeatureExtractor(data_path)
        self.features = None
        self.labels = None
        self.label_map = None
        self.model = None

        #METRICS
        self.training_losses = []
        self.training_accuracies = []
        self.training_precisions = []
        self.training_recalls = []

        self.validation_losses = []
        self.validation_accuracies = []
        self.validation_precisions = []
        self.validation_recalls = []

    # INITIALIZE MODEL
    def initialize_model(self, num_classes=None):
        info("Initializing ArcFace model...")

        self.extract_labels()

        if num_classes is None:
            if self.label_map:
                num_classes = len(self.label_map)

            if not num_classes:
                folders = [
                    d for d in os.listdir(self.data_path)
                    if os.path.isdir(os.path.join(self.data_path, d))
                ]
                num_classes = len(folders)
                info(f"Detected {num_classes} classes from dataset directory.")

        if num_classes == 0:
            raise ValueError("Number of classes is zero. Check dataset structure.")

        self.model = ArcFaceModel(
            feature_dim=512,
            num_classes=num_classes,
            model_dir=self.arcface_model_dir
        )

        info(f"ArcFace model initialized with {num_classes} classes.")

    # FEATURE & LABEL EXTRACTION
    def extract_labels(self):
        info("Extracting labels...")
        self.feature_extractor.extract_labels()
        self.label_map = self.feature_extractor.label_map
        info(f"Label map: {self.label_map}")

    def extract_features(self):
        info("Extracting features...")
        if self.model is None:
            self.initialize_model()

        self.feature_extractor.extract_features(self.model)
        self.features, self.labels = self.feature_extractor.get_features_and_labels()

        if self.features is None or len(self.features) == 0:
            raise ValueError("Feature extraction failed.")

        info("Feature extraction completed.")

    # TRAINING
    def train(self, num_epochs=50, lr=0.001, momentum=0.9, val_split_ratio=0.2):
        info("Starting ArcFace training...")

        if self.features is None or self.labels is None:
            self.extract_features()

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_split_ratio,
            random_state=42
        )

        train_idx, val_idx = next(sss.split(self.features, self.labels))
        train_features, val_features = self.features[train_idx], self.features[val_idx]
        train_labels, val_labels = self.labels[train_idx], self.labels[val_idx]

        info(f"Train samples: {len(train_features)} | Val samples: {len(val_features)}")

        trainer = ArcFaceTrainer(
            self.model,
            train_features,
            train_labels,
            lr=lr,
            momentum=momentum
        )

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            info(f"Epoch {epoch + 1}/{num_epochs}")

            #TRAIN
            train_loss, train_acc, train_prec, train_rec = trainer.train_epoch()
            self.training_losses.append(train_loss)
            self.training_accuracies.append(train_acc)
            self.training_precisions.append(train_prec)
            self.training_recalls.append(train_rec)

            #VALID
            val_loss, val_acc, val_prec, val_rec = trainer.evaluate_epoch(
                val_features, val_labels
            )
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_acc)
            self.validation_precisions.append(val_prec)
            self.validation_recalls.append(val_rec)

        
            info(
                f"[TRAIN] Loss: {train_loss:.4f} | "
                f"Acc: {train_acc:.4f} | "
                f"Prec: {train_prec:.4f} | "
                f"Rec: {train_rec:.4f}"
            )

            info(
                f"[VAL]   Loss: {val_loss:.4f} | "
                f"Acc: {val_acc:.4f} | "
                f"Prec: {val_prec:.4f} | "
                f"Rec: {val_rec:.4f}"
            )

            #SAVE BEST MODEL
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_save_path)
                info(f"✔ New best model saved (Val Acc = {best_val_acc:.4f})")

        info("Training completed.")

    # PLOT METRICS
    def plot_training_metrics(self, save_path=None):
        info("Plotting training metrics...")

        epochs = range(1, len(self.training_losses) + 1)
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.training_losses, label="Train Loss")
        plt.plot(epochs, self.validation_losses, label="Val Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.training_accuracies, label="Train Acc")
        plt.plot(epochs, self.validation_accuracies, label="Val Acc")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.training_precisions, label="Train Precision")
        plt.plot(epochs, self.validation_precisions, label="Val Precision")
        plt.title("Precision")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.training_recalls, label="Train Recall")
        plt.plot(epochs, self.validation_recalls, label="Val Recall")
        plt.title("Recall")
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            info(f"Metrics saved to {save_path}")
        else:
            plt.show()

    # LOAD MODEL
    def load_model(self):
        if not self.model_loaded:
            if self.model is None:
                self.initialize_model()
            self.model.load_state_dict(torch.load(self.model_save_path))
            self.model_loaded = True
            info(f"Model loaded from {self.model_save_path}")

    def model_exists(self):
        return os.path.exists(self.model_save_path)
