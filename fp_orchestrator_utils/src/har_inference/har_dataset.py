import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HARDataset(torch.utils.data.Dataset):
    """
    Human Activity Recognition (HAR) Dataset for variable-length sensor sequences.

    Handles multi-modal sensor data including IMU sensor and audio features with
    variable sequence lengths.

    Args:
        upload_features (list): List of dicts containing sensor data
        labels (np.ndarray): Array of activity labels
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, upload_features: list, labels: np.ndarray, transform=None):
        self.upload_features = upload_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.upload_features)

    def __getitem__(self, idx):
        upload_sample = self.upload_features[idx]

        sample = {
            'features': upload_sample['features'],
            'n_users': upload_sample['n_users'],
            'label': self.labels[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_sensor_info(self) -> dict:
        """
        Get sensor information and their shapes.
        """
        if len(self.upload_features) == 0:
            return {}
        
        sample_features = self.upload_features[0]['features']
        sensor_info = {}

        for sensor_type, data in sample_features.items():
            if isinstance(data, np.ndarray):
                sensor_info[sensor_type] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
            else: 
                sensor_info[sensor_type] = {
                    'type': str(type(data))
                }
        return sensor_info
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
def collate_variable_length(batch: list[dict]) -> tuple:
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
                padded_sequences = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)
                sensor_data['audio'] = padded_sequences
            else:
                # Variable-length sensors, pad sequences
                sequences = [torch.tensor(sample['features'][sensor_type], dtype=torch.float32) for sample in batch]
                padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
                sensor_data[sensor_type] = padded_sequences

        # Collect n_users and labels
        for item in batch:
            n_users_list.append(item['n_users'])
            labels_list.append(item['label'])

        n_users_tensor = torch.tensor(n_users_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long) if labels_list else None

        return sensor_data, n_users_tensor, labels_tensor

class HARDataModule:
    """
    Data module for HAR dataset with train, validation, and test splits and DataLoader.
    """
    def __init__(
            self,
            upload_features: list[dict],
            labels: np.ndarray,
            batch_size: int = 32,
            train_split: float = 0.8,
            val_split: float = 0.2,
            transform=None
    ):
        self.upload_features = upload_features
        self.labels = labels
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

        self.setup()

    def setup(self):
        """
        Setup train and validation datasets.
        """
        self.full_dataset = HARDataset(self.upload_features, self.labels, transform=self.transform)

        # Split dataset
        train_size = int(self.train_split * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, val_size])

        logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get DataLoader for training dataset.
        """
        if self.train_dataset is None:
            self.setup()

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get DataLoader for validation dataset.
        """
        if self.val_dataset is None:
            self.setup()

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_variable_length
        )
    
    def get_sensor_info(self) -> dict:
        """
        Get sensor information from the full dataset.
        """
        if self.full_dataset is None:
            self.setup()
        
        return self.full_dataset.get_sensor_info()

    def get_class_distribution(self) -> dict:
        """
        Get class distribution from the full dataset.
        """
        if self.full_dataset is None:
            self.setup()
        
        return self.full_dataset.get_class_distribution()