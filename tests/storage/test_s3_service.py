from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path

from fp_orchestrator_utils.storage.s3 import S3Service, S3Config


class TestS3Service:
    """Test cases for S3Service."""

    def test_s3_service_initialization(self, mock_s3_config, mock_s3_client):
        """Test S3Service initialization."""
        service = S3Service(mock_s3_config)
        
        assert service.bucket_name == mock_s3_config.bucket_name

    def test_is_connected_success(self, mock_s3_service):
        """Test successful S3 connection check."""
        mock_s3_service.client.head_bucket.return_value = {}
        
        result = mock_s3_service.is_connected()
        
        assert result is True
        mock_s3_service.client.head_bucket.assert_called_once_with(
            Bucket=mock_s3_service.bucket_name
        )

    def test_is_connected_client_error(self, mock_s3_service):
        """Test S3 connection check with ClientError."""
        mock_s3_service.client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'head_bucket'
        )
        
        result = mock_s3_service.is_connected()
        
        assert result is False

    def test_is_connected_no_credentials(self, mock_s3_service):
        """Test S3 connection check with NoCredentialsError."""
        mock_s3_service.client.head_bucket.side_effect = NoCredentialsError()
        
        result = mock_s3_service.is_connected()
        
        assert result is False

    def test_save_success(self, mock_s3_service):
        """Test successful data save to S3."""
        test_data = "test data content"
        test_key = "test/file.txt"
        
        mock_s3_service.client.put_object.return_value = {}
        
        result = mock_s3_service.save(test_data, test_key)
        
        assert result is True
        mock_s3_service.client.put_object.assert_called_once_with(
            Bucket=mock_s3_service.bucket_name,
            Key=test_key,
            Body=test_data
        )

    def test_save_client_error(self, mock_s3_service):
        """Test data save with ClientError."""
        test_data = "test data content"
        test_key = "test/file.txt"
        
        mock_s3_service.client.put_object.side_effect = ClientError(
            {'Error': {'Code': '403', 'Message': 'Forbidden'}}, 'put_object'
        )
        
        result = mock_s3_service.save(test_data, test_key)
        
        assert result is False

    def test_upload_success(self, mock_s3_service, sample_proto_file):
        """Test successful file upload to S3."""
        test_key = "test/sample.proto"
        
        mock_s3_service.client.upload_file.return_value = None
        
        result = mock_s3_service.save(sample_proto_file, test_key)
        
        assert result is True
        mock_s3_service.client.put_object.assert_called_once_with(
            Bucket=mock_s3_service.bucket_name,
            Key=test_key,
            Body=sample_proto_file
        )

    def test_download_success(self, mock_s3_service, temp_dir):
        """Test successful file download from S3."""
        test_key = "test/file.txt"
        local_path = temp_dir / "downloaded_file.txt"
        local_path = str(local_path)
        
        mock_s3_service.client.download_file.return_value = None
        
        result = mock_s3_service.download(test_key, local_path)
        
        assert result == local_path
        mock_s3_service.client.download_file.assert_called_once_with(
            mock_s3_service.bucket_name, test_key, str(local_path)
        )

    def test_download_client_error(self, mock_s3_service, temp_dir):
        """Test file download with ClientError."""
        test_key = "test/file.txt"
        local_path = temp_dir / "downloaded_file.txt"
        
        mock_s3_service.client.download_file.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'download_file'
        )
        
        try:
            mock_s3_service.download(test_key, local_path)
        except ClientError as e:
            assert str(e) == "An error occurred (404) when calling the download_file operation: Not Found"

    def test_list_objects_success(self, mock_s3_service):
        """Test successful object listing."""
        test_prefix = "test/"
        mock_response = {
            'Contents': [
                {'Key': 'test/file1.proto'},
                {'Key': 'test/file2.proto'},
                {'Key': 'test/subdir/file3.proto'}
            ]
        }
        
        mock_s3_service.client.list_objects_v2.return_value = mock_response
        
        result = mock_s3_service.list_objects(test_prefix)
        
        expected = ['test/file1.proto', 'test/file2.proto', 'test/subdir/file3.proto']
        assert result == expected
        mock_s3_service.client.list_objects_v2.assert_called_once_with(
            Bucket=mock_s3_service.bucket_name,
            Prefix=test_prefix
        )

    def test_list_objects_empty(self, mock_s3_service):
        """Test object listing with no contents."""
        test_prefix = "empty/"
        mock_response = {}  # No 'Contents' key
        
        mock_s3_service.client.list_objects_v2.return_value = mock_response
        
        result = mock_s3_service.list_objects(test_prefix)
        
        assert result == []
