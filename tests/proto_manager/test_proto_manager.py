import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from fp_orchestrator_utils.proto_manager.proto_manager import ProtoManager


class TestProtoManager:
    """Test cases for ProtoManager."""

    def test_proto_manager_initialization(self, mock_environment_variables, mock_s3_client):
        """Test ProtoManager initialization with environment variables."""
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service'):
            manager = ProtoManager()
            
            assert manager.proto_prefix == "proto/"
            assert manager.proto_local_dir == Path("tests/fixtures/proto")
            assert manager.grpc_output_dir == Path("tests/fixtures/grpc")

    def test_proto_manager_missing_env_vars(self):
        """Test ProtoManager initialization fails with missing environment variables."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                ProtoManager()

    def test_check_connection_success(self, mock_environment_variables, mock_s3_client):
        """Test successful connection check."""
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service') as mock_service_class:
            mock_service = MagicMock()
            mock_service.is_connected.return_value = True
            mock_service_class.return_value = mock_service
            
            manager = ProtoManager()
            result = manager.check_connection()
            
            assert result is True
            mock_service.is_connected.assert_called_once()

    def test_check_connection_failure(self, mock_environment_variables, mock_s3_client):
        """Test failed connection check."""
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service') as mock_service_class:
            mock_service = MagicMock()
            mock_service.is_connected.return_value = False
            mock_service_class.return_value = mock_service
            
            manager = ProtoManager()
            result = manager.check_connection()
            
            assert result is False

    def test_download_protos_success(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test successful proto files download."""
        mock_objects = [
            "service1.proto",
            "service2.proto",
            "not_proto.txt"  # Should be skipped
        ]
        
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service') as mock_service_class:
            mock_service = MagicMock()
            mock_service.list_objects.return_value = mock_objects
            mock_service.download.return_value = True
            mock_service_class.return_value = mock_service
            
            # Override the proto_local_dir to use temp directory
            with patch.object(Path, 'mkdir'):
                manager = ProtoManager()
                manager.proto_local_dir = temp_dir
                
                result = manager.download_protos()
                result = [Path(file) for file in result]
                
                # Should download 2 .proto files, skip the .txt file
                assert len(result) == 2
                assert Path(temp_dir / "service1.proto") in result
                assert Path(temp_dir / "service2.proto") in result

                # Verify S3 service calls
                mock_service.list_objects.assert_called_once_with(prefix="proto/")
                assert mock_service.download.call_count == 2

    def test_upload_protos_success(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test successful proto files upload."""
        # Create test proto files
        proto_file1 = temp_dir / "service1.proto"
        proto_file2 = temp_dir / "service2.proto"
        proto_file1.write_text("syntax = 'proto3';")
        proto_file2.write_text("syntax = 'proto3';")
        
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service') as mock_service_class:
            mock_service = MagicMock()
            mock_service.save.return_value = True
            mock_service_class.return_value = mock_service
            
            with patch.object(Path, 'mkdir'):
                manager = ProtoManager()
                manager.proto_local_dir = temp_dir
                
                manager.upload_protos()
                
                # Should save both proto files
                assert mock_service.save.call_count == 2
                
                # Verify the calls were made with correct arguments
                expected_calls = [
                    call(proto_file1.read_bytes(), "proto/service1.proto"),
                    call(proto_file2.read_bytes(), "proto/service2.proto")
                ]
                mock_service.save.assert_has_calls(expected_calls, any_order=True)

    def test_upload_protos_specific_files(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test upload of specific proto files."""
        proto_file = temp_dir / "specific.proto"
        proto_file.write_text("syntax = 'proto3';")
        
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service') as mock_service_class:
            mock_service = MagicMock()
            mock_service.upload.return_value = True
            mock_service_class.return_value = mock_service
            
            with patch.object(Path, 'mkdir'):
                manager = ProtoManager()

                manager.upload_protos([str(proto_file)])

                # Upload file content
                mock_service.save.assert_called_once_with(
                    proto_file.read_bytes(),
                    "proto/specific.proto"
                )

    def test_generate_grpc_code_success(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test successful gRPC code generation."""
        # Create test proto files
        proto_file1 = temp_dir / "service1.proto"
        proto_file2 = temp_dir / "service2.proto"
        proto_file1.write_text("syntax = 'proto3';")
        proto_file2.write_text("syntax = 'proto3';")
                
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service'):
            with patch.object(Path, 'mkdir'):
                manager = ProtoManager()
                manager.proto_local_dir = temp_dir
                manager.grpc_output_dir = temp_dir / "grpc"
                
                result = manager.generate_grpc_code()
                
                # Should process both files
                assert len(result) == 2
                assert str(proto_file1) in result
                assert str(proto_file2) in result
                

    def test_generate_grpc_code_failure(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test gRPC code generation with subprocess failure."""
        proto_file = temp_dir / "service.proto"
        proto_file.write_text("syntax = 'proto3';")
        
        # Mock failure on os.system call
        with patch('fp_orchestrator_utils.proto_manager.proto_manager.os.system') as mock_os_system:
            mock_os_system.side_effect = Exception("Command failed")
        
            with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service'):
                with patch.object(Path, 'mkdir'):
                    manager = ProtoManager()
                    manager.proto_local_dir = temp_dir
                    
                    result = manager.generate_grpc_code()
                    
                    # Should return empty list due to failure
                    assert result == []

    def test_generate_grpc_code_specific_files(self, mock_environment_variables, mock_s3_client, temp_dir):
        """Test gRPC code generation for specific files."""
        proto_file = temp_dir / "specific.proto"
        proto_file.write_text("syntax = 'proto3';")

        with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service'):
            with patch.object(Path, 'mkdir'):
                manager = ProtoManager()
                manager.proto_local_dir = temp_dir
                manager.grpc_output_dir = temp_dir / "grpc"

                result = manager.generate_grpc_code([str(proto_file)])
                
                assert result == [str(proto_file)]
