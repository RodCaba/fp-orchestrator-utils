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

    def load_data_from_s3(self) -> list[str]:
        """
        Loads data files from S3 and saves them to the local directory.
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
                    if key.endswith('.json'):
                      content = json.loads(self.s3_service.load(key))
                      data.append(content)
                      progress_bar.update(1)
            progress_bar.close()
            return data

        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            return data
        
    def preprocess_data(self, data: list[dict]) -> Tuple[Dict, np.array]:
        """
        Preprocesses raw data from S3 to features and labels arrays.
        """
        accelerometer_features = []
        gyroscope_features = []
        total_acceleration_features = []
        gravity_features = []
        orientation_features = []
        audio_features = []
        labels = []
        n_users_list = []

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

            for item in batch_data:
                # Skip bad tagged IMU data
                if item['sensor_type'] == 'imu':
                    continue
                elif item['sensor_type'] == 'audio':
                    if 'data' not in item or 'features' not in item['data']:
                        continue
                    audio_batch = self._process_audio_batch(item)
                    logger.info(audio_batch)

                    if audio_batch is not None:
                        audio_features.append(audio_batch)
                    else:
                        audio_features.append(np.zeros((64, 126)))

                else:
                    imu_batch = self._process_imu_batch(item)

                    if imu_batch is not None:
                        if item['sensor_type'] == 'accelerometer':
                            accelerometer_features.append(imu_batch)
                        elif item['sensor_type'] == 'gyroscope':
                            gyroscope_features.append(imu_batch)
                        elif item['sensor_type'] == 'gravity':
                            gravity_features.append(imu_batch)
                        elif item['sensor_type'] == 'totalacceleration':
                            total_acceleration_features.append(imu_batch)
                        elif item['sensor_type'] == 'orientation':
                            orientation_features.append(imu_batch)
                    else:
                        # Append zero array if data is missing or malformed
                        if item['sensor_type'] == 'accelerometer':
                            accelerometer_features.append(np.zeros((100, 3)))
                        elif item['sensor_type'] == 'gyroscope':
                            gyroscope_features.append(np.zeros((100, 3)))
                        elif item['sensor_type'] == 'gravity':
                            gravity_features.append(np.zeros((100, 3)))
                        elif item['sensor_type'] == 'totalacceleration':
                            total_acceleration_features.append(np.zeros((100, 3)))
                        elif item['sensor_type'] == 'orientation':
                            orientation_features.append(np.zeros((100, 7)))

                # Store label and n_users
                labels.append(upload_label)
                n_users_list.append(n_users)

                progress_bar.update(1)

        # Convert lists to numpy arrays
        accelerometer_features = np.array(accelerometer_features)
        gyroscope_features = np.array(gyroscope_features)
        total_acceleration_features = np.array(total_acceleration_features)
        gravity_features = np.array(gravity_features)
        orientation_features = np.array(orientation_features)
        audio_features = np.array(audio_features)
        n_users_array = np.array(n_users_list)

        # Encode labels
        labels_encoded = LabelEncoder().fit_transform(labels)

        features = {
            'accelerometer': accelerometer_features,
            'gyroscope': gyroscope_features,
            'total_acceleration': total_acceleration_features,
            'gravity': gravity_features,
            'orientation': orientation_features,
            'audio': audio_features,
            'n_users': n_users_array
        }

        return features, labels_encoded

    def _process_imu_batch(self, imu_data: dict) -> np.ndarray:
        """
        Processes a batch of IMU data into a numpy array.
        """
        if not imu_data:
            return None
        
        processed = []
        data = imu_data.get('data', {})
        if 'x' in data and 'y' in data and 'z' in data:
            processed.append([data['x'], data['y'], data['z']])
        elif 'qx' in data:
            processed.append([
                data.get('qx', 0.0),
                data.get('qy', 0.0),
                data.get('qz', 0.0),
                data.get('qw', 1.0),  # Default to 1.0 for qw if not present
                data.get('roll', 0.0),
                data.get('pitch', 0.0),
                data.get('yaw', 0.0)
            ])

        if not processed:
            return None

        # Convert list of IMU data to numpy array
        imu_array = np.array(processed)

        return imu_array
    
    def _process_audio_batch(self, audio_data: dict) -> np.ndarray:
        """
        Process audio features from a batch
        """
        if not audio_data:
            return None
        
        processed = []

        data = audio_data.get('data', {})
        features = data.get('features', [])
        feature_data = features.get('feature_data', [])
        processed.append(feature_data)

        if not processed:
            return None

        feature_array = np.array(processed)

        if len(feature_array.shape) == 3 and feature_array.shape[0] == 1:
            # Shape (1, time_steps, n_mels)
            feature_array = feature_array.squeeze(0)  # Remove batch dimension
        elif len(feature_array.shape) == 1:
            n_mels = features.get('feature_parameters', {}).get('n_mels', 126)
            if len(feature_array) % n_mels != 0:
                logger.warning("Feature data length is not a multiple of n_mels")
                return None
            time_steps = len(feature_array) // n_mels
            feature_array = feature_array.reshape((time_steps, n_mels))
        else:
            logger.warning("Unexpected feature array shape: {}".format(feature_array.shape))
            return None

        return feature_array