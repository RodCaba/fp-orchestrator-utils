from fp_orchestrator_utils.storage import S3Service, S3Config
import os
import logging
import json
from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tqdm

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.config = S3Config(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            bucket_name=os.getenv("S3_BUCKET_NAME", ""),
            region=os.getenv("AWS_REGION", "")

        )
        self.s3_service = S3Service(self.config)
        self.data_prefix = os.getenv("S3_DATA_PREFIX", "")

    def load_data_from_s3(self, save_locally: bool = False) -> list[str]:
        """
        Loads data files from S3 and saves them to the local directory.
        :param save_locally: Flag to indicate whether to save files locally.
        :return: The raw data files downloaded from S3.
        """
        data = []

        try:
            paginator = self.s3_service.get_paginator(self.data_prefix)
            total_files = self.s3_service.count_objects(self.data_prefix)
            progress_bar = tqdm.tqdm(total=total_files, desc="Loading data from S3")
            for page in paginator:

                for obj in page.get('Contents', []):
                    key = obj['Key']
                    print(f"Processing key: {key}")  # Debugging line
                    if key.endswith('.json'):
                      content = json.loads(self.s3_service.load(key))
                      if save_locally:
                          with open(key, 'w') as f:
                              json.dump(content, f)
                      data.append(content)
                      progress_bar.update(1)
            progress_bar.close()
            return data

        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            return data
    
    def load_local_data(self) -> list[str]:
        """
        Loads data files from a local directory.
        :return: The raw data files loaded from the local directory.
        """
        data = []
        directory = self.data_prefix if self.data_prefix else 'orchestrator_data/'
        try:
            progress_bar = tqdm.tqdm(desc="Loading data from local directory")
            total_files = len(os.listdir(directory))
            progress_bar.total = total_files
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    with open(os.path.join(directory, filename), 'r') as f:
                        content = json.load(f)
                        data.append(content)
                    progress_bar.update(1)
            progress_bar.close()
            if not data:
                raise FileNotFoundError("No JSON files found in directory")
            return data
        except Exception as e:
            logger.error(f"Failed to load local data: {e}")
            return data
        
    def preprocess_data(self, data: list[dict]) -> Tuple[Dict, np.array]:
        """
        Preprocesses raw data from S3 to features and labels arrays.
        It will preprocess data for variable-length sequences.
        """
        upload_features = []
        labels = []

        total_files = len(data)
        progress_bar = tqdm.tqdm(total=total_files, desc="Preprocessing data")

        for upload in data:
            upload_label = upload.get('label', None)
            n_users = upload.get('n_users', 0)

            if upload_label is None:
                continue

            batch_data = upload.get('data', [])

            if not batch_data:
                continue

            sensor_sequences = {
                'accelerometer': [],
                'gyroscope': [],
                'gravity': [],
                'totalacceleration': [],
                'orientation': [],
                'audio': []
            }

            for item in batch_data:
                # Skip bad tagged IMU data
                sensor_type = item.get('sensor_type', None)

                if (sensor_type == 'imu' or
                    sensor_type == 'accelerometeruncalibrated' or
                    sensor_type == 'gyroscopeuncalibrated'
                    ):
                    continue

                elif sensor_type == 'audio':
                    if 'data' not in item or 'features' not in item['data']:
                        continue
                    audio_batch = self._process_audio_batch(item)

                    if audio_batch is not None:
                        sensor_sequences['audio'].append(audio_batch)

                else:
                    imu_batch = self._process_imu_batch(item)

                    if imu_batch is not None:
                        sensor_sequences[sensor_type].append(imu_batch)
            
            # Convert lists to numpy arrays, handling empty cases
            for sensor_type, sequence in sensor_sequences.items():
                if sensor_type == 'audio' and sequence:
                    sensor_sequences['audio'] = [np.array(seq) for seq in sequence]
                elif sensor_type == 'audio':
                    sensor_sequences['audio'] = [np.zeros((1, 126))]
                elif sensor_type != 'audio' and sequence:
                    sensor_sequences[sensor_type] = np.array(sequence)
                elif sensor_type != 'audio':
                    if sensor_type == 'orientation':
                        sensor_sequences[sensor_type] = np.zeros((1, 7))
                    else:
                        sensor_sequences[sensor_type] = np.zeros((1, 3))
            
            # Handle missing audio
            if not sensor_sequences['audio']:
                sensor_sequences['audio'] = np.zeros((64, 126))

            upload_sample = {
                'features': sensor_sequences,
                'label': upload_label,
                'n_users': n_users
            }

            upload_features.append(upload_sample)
            labels.append(upload_label)
            progress_bar.update(1)

        # Encode labels
        labels_encoded = LabelEncoder().fit_transform(labels)

        with open('features_labels_debug.json', 'w') as f:
            json.dump({'features': [str(f['features']) for f in upload_features], 'labels': labels_encoded.tolist()}, f)

        return upload_features, labels_encoded

    def _process_imu_batch(self, imu_data: dict) -> np.ndarray:
        """
        Processes a batch of IMU data into a numpy array.
        """
        if not imu_data:
            return None
        
        data = imu_data.get('data', {})
        if 'x' in data and 'y' in data and 'z' in data:
            return np.array([data['x'], data['y'], data['z']])
        elif 'qx' in data:
            return np.array([
                data.get('qx', 0.0),
                data.get('qy', 0.0),
                data.get('qz', 0.0),
                data.get('qw', 1.0),
                data.get('roll', 0.0),
                data.get('pitch', 0.0),
                data.get('yaw', 0.0)
            ])

        return None
    
    def _process_audio_batch(self, audio_data: dict) -> np.ndarray:
        """
        Process audio features from a batch
        """
        if not audio_data:
            return None
        
        data = audio_data.get('data', {})
        features = data.get('features', [])
        feature_data = features.get('feature_data', [])

        if not feature_data:
            return None

        feature_array = np.array(feature_data)

        if len(feature_array.shape) == 1:
            n_mels = features.get('feature_parameters', {}).get('n_mels', 126)
            if len(feature_array) % n_mels != 0:
                logger.warning("Feature data length is not a multiple of n_mels")
                return None
            time_steps = len(feature_array) // n_mels
            feature_array = feature_array.reshape((time_steps, n_mels))
        elif len(feature_array.shape) == 2:
            pass  # Already in (time_steps, n_mels) shape
        else:
            logger.warning("Unexpected feature array shape: {}".format(feature_array.shape))
            return None

        return feature_array