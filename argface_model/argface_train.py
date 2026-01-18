import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from logger import info

class ArcFaceTrainer:
    def __init__(self, model, features, labels, lr=0.01, momentum=0.9):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.features = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
            self.model.train()
            self.optimizer.zero_grad()
            
            # 1. Chạy mô hình để tính toán outputs trước
            outputs = self.model(self.features)
            
            # 2. Đặt lệnh print debug ngay SAU khi có outputs và TRƯỚC khi tính loss
            print(f"Shape outputs: {outputs.shape}") 
            print(f"Shape labels: {self.labels.shape}") 
            print(f"Type labels: {self.labels.dtype}") 
            
            # 3. Tính loss
            loss = self.criterion(outputs, self.labels)
            
            loss.backward()
            self.optimizer.step()

            # Tính toán các metrics
            _, predicted = torch.max(outputs, 1)
            labels_np = self.labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            accuracy = accuracy_score(labels_np, predicted_np)
            precision = precision_score(labels_np, predicted_np, average='weighted', zero_division=0)
            recall = recall_score(labels_np, predicted_np, average='weighted', zero_division=0)

            # Trả về 4 giá trị
            return loss.item(), accuracy, precision, recall
        
    def get_predictions(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.features)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate_epoch(self, val_features, val_labels):
        self.model.eval()
        total_loss = 0
        all_preds = []
        
        val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        val_dataset = torch.utils.data.TensorDataset(val_features_tensor, val_labels_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(val_dataset.tensors[0])
        
        # Tính toán các metrics
        accuracy = accuracy_score(val_labels, all_preds)
        precision = precision_score(val_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(val_labels, all_preds, average='weighted', zero_division=0)
        
        self.model.train()
        # Trả về 4 giá trị
        return avg_loss, accuracy, precision, recall