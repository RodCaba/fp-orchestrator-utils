from ..abstractions import BaseStorage
import boto3
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Config:
    """
    Configuration class for S3 service.
    """
    def __init__(
            self,
            access_key: str,
            secret_key: str,
            bucket_name: str,
            region: str = 'us-east-2'
          ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.region = region

class S3Service(BaseStorage):
    """
    S3 storage implementation of the BaseStorage interface.
    """
    client: boto3.client
    def __init__(self, config: S3Config):
        self.client: boto3.client = boto3.client(
            's3',
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region
        )
        self.bucket_name = config.bucket_name

    def save(self, data, key: str):
        """
        Save data to S3 bucket.
        
        :param data: Data to be saved.
        :param key: Key under which the data will be stored.
        """
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data
            )
            return True
        except ClientError as e:
            logger.error(f"Failed to save data to S3: {e}")
            return False

    def load(self, key: str):
        """
        Load data from S3 bucket using a key.
        
        :param key: Key for the data to be loaded.
        :return: Loaded data.
        """
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()
    
    def download(self, key: str, file_path: str):
        """
        Download data from S3 bucket using a key.
        
        :param key: Key for the data to be downloaded.
        :param file_path: Local file path to save the downloaded data.
        :return: Path to the downloaded file.
        """
        try:
            self.client.download_file(self.bucket_name, key, file_path)
        except ClientError as e:
            logger.error(f"Failed to download {key} from S3: {e}")
            return False
        return file_path
    
    def list_objects(self, prefix: str = ''):
        """
        List objects in the S3 bucket in a directory.

        :param prefix: Prefix to filter the objects.
        :return: List of object keys in the bucket.
        """
        logger.info(f"Listing objects with prefix: {prefix}")
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]
    
    def count_objects(self, prefix: str = '') -> int:
        """
        Count objects in the S3 bucket with a given prefix.

        :param prefix: Prefix to filter the objects.
        :return: Number of objects in the bucket with the given prefix.
        """
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return len(response.get('Contents', []))
    
    def get_paginator(self, prefix: str = ''):
        """
        Get a paginator for listing objects in the S3 bucket.

        :param prefix: Prefix to filter the objects.
        :return: Paginator object.
        """
        return self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket_name, Prefix=prefix)

    def is_connected(self):
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
