from storage.s3 import S3Config, S3Service
import os
import logging
import json
from typing import Dict, Tuple
import numpy as np

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

        # Ensure local directory exists
        os.makedirs(self.data_local_dir, exist_ok=True)

    def load_data_from_s3(self) -> list[str]:
        """
        Loads data files from S3 and saves them to the local directory.
        :return: The raw data files downloaded from S3.
        """
        data = []

        try:
            paginator = self.s3_service.get_paginator(self.data_prefix)
            for page in paginator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.json'):
                      content = json.loads(self.s3_service.load(key))
                      data.append(content)
            return data

        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            return data
        
    def preprocess_data(self, data: list[dict]) -> Tuple[Dict, np.array]:
        """
        Preprocesses raw data from S3 to features and labels arrays.
        """
        imu_features = []
        audio_features = []
        labels = []

        batches = {}
        for item in data:
            # Skip IMU data bad tagged.
            if item['sensor_type'] == 'imu':
                continue
            batch_id = item.get('batch_id', 'default')

            if batch_id not in batches:
                batches[batch_id] = {
                    'imu': [],
                    'audio': [],
                    'labels': []
                }
            
            if item['sensor_type'] == 'audio':
                batches[batch_id]['audio'].append(item['data'])
            else:
                batches[batch_id]['imu'].append(item['data'])