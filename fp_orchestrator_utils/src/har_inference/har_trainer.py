from .har_model import HARModel
import torch
import torch.nn as nn
import numpy as np
import logging
import os
from fp_orchestrator_utils.storage.s3 import S3Service, S3Config

logger = logging.getLogger(__name__)

class HARTrainer:
    def __init__(self, model: HARModel, device: str = 'cpu'):
        self.model = model
        self.device = device
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

        dataset = VariableLengthDataset(upload_features, labels)

        # Split into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_datasert, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(train_datasert, batch_size=32, shuffle=True, collate_fn=self.collate_variable_length)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_variable_length)

        return train_loader, val_loader
    
    def collate_variable_length(self, batch):
        """
        Custom collate function to handle variable-length sequences.
        """
        sensor_data = {}
        n_users_list = []
        labels_list = []

        # Get all sensor types from the first sample
        sample_sensors = batch[0]['features'].keys()

        for sensor_type in sample_sensors:
            if sensor_type == 'audio':
                audio_tensors = [torch.tensor(sample['features']['audio'], dtype=torch.float32) for sample in batch]
                padded_sequences = nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)
                sensor_data['audio'] = padded_sequences
            else:
                # Variable-length sensors, pad sequences
                sequences = [torch.tensor(sample['features'][sensor_type], dtype=torch.float32) for sample in batch]
                padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
                sensor_data[sensor_type] = padded_sequences

        # Collect n_users and labels
        for item in batch:
            n_users_list.append(item['n_users'])
            labels_list.append(item['label'])

        n_users_tensor = torch.tensor(n_users_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return sensor_data, n_users_tensor, labels_tensor

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
        val_correct = 0
        val_total = 0

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
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        logger.info(f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
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
        # Create dummy inputs with correct shapes
        dummy_sensor_data = {
            'accelerometer': torch.randn(batch_size, seq_length, 3),
            'gyroscope': torch.randn(batch_size, seq_length, 3),
            'totalacceleration': torch.randn(batch_size, seq_length, 3),
            'gravity': torch.randn(batch_size, seq_length, 3),
            'orientation': torch.randn(batch_size, seq_length, 7),
            'audio': torch.randn(batch_size, 5, 64, 126)
        }
        dummy_n_users = torch.tensor([1.0], dtype=torch.float32)
        # Move to device
        for sensor_type in dummy_sensor_data:
            dummy_sensor_data[sensor_type] = dummy_sensor_data[sensor_type].to(self.device)
        dummy_n_users = dummy_n_users.to(self.device)

        torch.onnx.export(
            self.model,
            (dummy_sensor_data, dummy_n_users),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['sensor_data', 'n_users'],
            output_names=['output'],
            dynamic_axes={'sensor_data': {1: 'sequence_length'},
                          'output': {}}
        )
        logger.info(f"Model exported to {onnx_path}")

        try:
            self.s3_service.upload(onnx_path, self.model_prefix + onnx_path)
            logger.info("Uploaded ONNX model to S3.")
        except Exception as e:
            logger.error(f"Failed to upload ONNX model to S3: {e}")
        logger.info("Export to ONNX completed.")

class VariableLengthDataset(torch.utils.data.Dataset):
    def __init__(self, upload_features: list, labels: np.ndarray):
        self.upload_features = upload_features
        self.labels = labels

    def __len__(self):
        return len(self.upload_features)

    def __getitem__(self, idx):
        upload_sample = self.upload_features[idx]

        return {
            'features': upload_sample['features'],
            'n_users': upload_sample['n_users'],
            'label': self.labels[idx]
        }