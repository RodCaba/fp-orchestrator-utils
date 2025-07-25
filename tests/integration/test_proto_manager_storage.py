import pytest
from unittest.mock import patch, MagicMock
from fp_orchestrator_utils import ProtoManager, S3Config, S3Service

class TestProtoManagerStorage:
    """Integration tests for ProtoManager S3 storage."""

    @pytest.fixture
    def mock_complete_environment(self):
        """ Complete mock environment for integration tests. """
        env_vars = {
            "AWS_ACCESS_KEY_ID": "mock_access_key",
            "AWS_SECRET_ACCESS_KEY": "mock_secret_key",
            "AWS_REGION": "us-east-1",
            "S3_BUCKET_NAME": "mock_bucket_name",
            "S3_PROTO_PREFIX": "proto/",
            "PROTO_LOCAL_DIR": "./tests/fixtures/proto",
            "PROTO_GRPC_OUTPUT_DIR": "./tests/fixtures/grpc"
        }

        with patch.dict('os.environ', env_vars):
            with patch('boto3.client') as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance

                # Mock successful S3 ops
                mock_instance.head_bucket.return_value = {}
                mock_instance.list_objects_v2.return_value = {
                    'Contents': [
                        {'Key': 'proto/service1.proto'},
                        {'Key': 'proto/service2.proto'},
                        {'Key': 'proto/not_proto.txt'}
                    ]
                }
                mock_instance.download_file.return_value = ""
                mock_instance.put_object.return_value = {}

                yield mock_instance

    def test_complete_proto_flow(self, mock_complete_environment):
        """Test complete flow of downloading and uploading proto files."""
        # Initialize ProtoManager
        manager = ProtoManager()

        # Check connection
        assert manager.check_connection() is True

        # Download protos
        protos = downloaded_files = manager.download_protos()
        assert len(downloaded_files) == 2

        # Generate gRPC code
        grpc_files = manager.generate_grpc_code(protos)
        assert len(grpc_files) == 2

        # Upload protos
        manager.upload_protos()

        # Verify S3 service calls
        mock_complete_environment.list_objects_v2.assert_called_once_with(Bucket="mock_bucket_name", Prefix="proto/")
        mock_complete_environment.head_bucket.assert_called_once_with(Bucket="mock_bucket_name")

    def test_error_handling(self, mock_complete_environment):
        """Test error handling in ProtoManager methods."""
        # Mock S3Service to raise an exception
        mock_complete_environment.head_bucket.side_effect = Exception("S3 connection error")

        manager = ProtoManager()

        # Check connection should fail
        assert manager.check_connection() is False
