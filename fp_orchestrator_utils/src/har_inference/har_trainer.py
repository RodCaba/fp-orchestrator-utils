from .har_model import HARModel
import torch
import torch.nn as nn
import numpy as np
import logging
import os
from fp_orchestrator_utils.storage.s3 import S3Service, S3Config
from .har_dataset import HARDataModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class ONNXModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, accelerometer, gyroscope, totalacceleration, gravity, orientation, audio, n_users):
                sensor_data = {
                    'accelerometer': accelerometer,
                    'gyroscope': gyroscope,
                    'totalacceleration': totalacceleration,
                    'gravity': gravity,
                    'orientation': orientation,
                    'audio': audio
                }
                return self.model(sensor_data, n_users)

class HARTrainer:
    def __init__(self, model: HARModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.model_prefix = os.getenv("S3_MODEL_PREFIX", "har_model/")

        s3_config = S3Config(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            bucket_name=os.getenv("S3_BUCKET_NAME", ""),
        )
        self.s3_service = S3Service(s3_config)

    def prepare_data(self, upload_features: list, labels: np.ndarray) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Prepares DataLoader for training and validation with variable-length sequences.
        :param upload_features: List of dictionaries with 'features', 'label', and 'n_users' keys.
        """
        logger.info(f"Preparing data with {len(upload_features)} samples")
        
        data_module = HARDataModule(
            upload_features,
            labels,
            batch_size=32,
            train_split=0.8,
            val_split=0.2,
        )


        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        logger.info(f"Sensor info: {data_module.get_sensor_info()}")
        logger.info(f"Class distribution: {data_module.get_class_distribution()}")

        return train_loader, val_loader

    def load_checkpoint(self, checkpoint_path: str = 'best_har_model.pth') -> bool:
        """
        Loads model checkpoint.
        """
        try:
            # Loads the checkpoint from S3
            downloaded = self.s3_service.download(self.model_prefix + 'best_har_model.pth', checkpoint_path)
            if not downloaded:
                logger.warning("No checkpoint found in S3.")
                return False
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint file {checkpoint_path} does not exist.")
                return False
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def evaluate_model(self, val_loader: torch.utils.data.DataLoader):
        """
        Evaluate current model an return accuracy and loss
        """
        self.model.eval()
        val_loss = 0.0
        all_predicted = []
        all_labels = []

        with torch.no_grad():
            for sensor_data, n_users, labels in val_loader:
                for sensor_type in sensor_data:
                    sensor_data[sensor_type] = sensor_data[sensor_type].to(self.device)
                n_users = n_users.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sensor_data, n_users)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        all_predicted = np.array(all_predicted)
        all_labels = np.array(all_labels)

        # Basic metrics
        val_acc = accuracy_score(all_labels, all_predicted) * 100
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        # Detailed metrics
        val_precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_predicted)
        class_report = classification_report(all_labels, all_predicted)

        logger.info(f"Validation Metrics: \n")
        logger.info(f"Accuracy: {val_acc:.2f}% \n")
        logger.info(f"Precision: {val_precision:.4f} \n")
        logger.info(f"Recall: {val_recall:.4f} \n")
        logger.info(f"F1 Score: {val_f1:.4f} \n")
        logger.info(f"Confusion Matrix: \n{cm} \n")
        logger.info(f"Classification Report: \n{class_report} \n")


        return val_acc, avg_val_loss

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int = 50,
            resume_from_checkpoint: bool = True,
        ):
        """
        Trains the HAR model.
        """
        # Load last model if available
        if resume_from_checkpoint:
            if self.load_checkpoint():
                logger.info("Evaluating loaded model before training...")
                current_val_acc, _ = self.evaluate_model(val_loader)
                self.best_val_acc = current_val_acc
                logger.info(f"Starting training with best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                logger.info("No checkpoint found, starting training from scratch.")

        for epoch in range(epochs):
            try:
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for sensor_data, n_users, labels in train_loader:
                    for sensor_type in sensor_data:
                        sensor_data[sensor_type] = sensor_data[sensor_type].to(self.device)
                    n_users = n_users.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(sensor_data, n_users)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                val_acc, val_loss = self.evaluate_model(val_loader)

                train_acc = 100 * train_correct / train_total

                logger.info(f'Epoch [{epoch+1}/{epochs}], '
                            f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, '
                            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'best_har_model.pth')
                    logger.info(f"Saved best model with Val Acc: {self.best_val_acc:.2f}%")
            except Exception as e:
                logger.error(f"Error during training at epoch {epoch+1}: {e}")
                continue
        # Upload the best model to S3
        try:
            self.s3_service.upload('best_har_model.pth', self.model_prefix + 'best_har_model.pth')
            logger.info("Uploaded best model to S3.")
        except Exception as e:
            logger.error(f"Failed to upload best model to S3: {e}")
        logger.info("Training completed.")

    def export_to_onnx(self, onnx_path: str = 'har_model.onnx'):
        """
        Exports the trained model to ONNX format.
        """
        self.model.eval()
        batch_size = 1
        seq_length = 50
        
        # Create dummy inputs with correct shapes for each sensor
        dummy_accelerometer = torch.randn(batch_size, seq_length, 3).to(self.device)
        dummy_gyroscope = torch.randn(batch_size, seq_length, 3).to(self.device)
        dummy_totalacceleration = torch.randn(batch_size, seq_length, 3).to(self.device)
        dummy_gravity = torch.randn(batch_size, seq_length, 3).to(self.device)
        dummy_orientation = torch.randn(batch_size, seq_length, 7).to(self.device)
        dummy_audio = torch.randn(batch_size, 5, 64, 126).to(self.device)
        dummy_n_users = torch.tensor([1.0], dtype=torch.float32).to(self.device)

        wrapper_model = ONNXModelWrapper(self.model).to(self.device)

        torch.onnx.export(
            wrapper_model,
            (dummy_accelerometer, dummy_gyroscope, dummy_totalacceleration, 
             dummy_gravity, dummy_orientation, dummy_audio, dummy_n_users),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['accelerometer', 'gyroscope', 'totalacceleration', 
                        'gravity', 'orientation', 'audio', 'n_users'],
            output_names=['output'],
            dynamic_axes={
                'accelerometer': {1: 'sequence_length'},
                'gyroscope': {1: 'sequence_length'}, 
                'totalacceleration': {1: 'sequence_length'},
                'gravity': {1: 'sequence_length'},
                'orientation': {1: 'sequence_length'},
                'audio': {2: 'audio_sequence_length'},
                'output': {}
            }
        )
        logger.info(f"Model exported to {onnx_path}")

        try:
            self.s3_service.upload(onnx_path, self.model_prefix + onnx_path)
            logger.info("Uploaded ONNX model to S3.")
        except Exception as e:
            logger.error(f"Failed to upload ONNX model to S3: {e}")
        logger.info("Export to ONNX completed.")
