import pytest
import tempfile
from pathlib import Path
import shutil
import os
from unittest.mock import patch

from fp_orchestrator_utils.storage.s3 import S3Service, S3Config
from fp_orchestrator_utils.proto_manager import ProtoManager

@pytest.fixture
def temp_dir():
    """ Fixture to create a temporary directory for testing. """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_proto_content():
    """ Fixture to provide sample content for .proto files. """
    return """
syntax = "proto3";

package test;

service TestService {
  rpc TestMethod (TestRequest) returns (TestResponse);
}

message TestRequest {
  string message = 1;
}

message TestResponse {
  string reply = 1;
}
"""

@pytest.fixture
def sample_proto_file(temp_dir, sample_proto_content):
    """ Fixture to create a sample .proto file in the temporary directory. """
    proto_file = temp_dir / "test.proto"
    proto_file.write_text(sample_proto_content)
    return proto_file

@pytest.fixture
def mock_s3_config():
    """ Fixture to provide a mock s3 config """
    return S3Config(
        access_key="mock_access_key",
        secret_key="mock_secret_key",
        bucket_name="mock_bucket_name",
        region="us-east-2"
    )

@pytest.fixture
def mock_s3_client():
    """ Fixture to create a mock S3 client. """
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def mock_s3_service(mock_s3_config, mock_s3_client):
    """ Fixture to create a mock S3Service instance. """
    service = S3Service(mock_s3_config)
    return service

@pytest.fixture
def mock_environment_variables():
    """ Fixture to set mock environment variables for testing. """
    env_vars = {
        "AWS_ACCESS_KEY_ID": "mock_access_key",
        "AWS_SECRET_ACCESS_KEY": "mock_secret_key",
        "AWS_S3_BUCKET": "mock_bucket_name",
        "AWS_REGION": "us-east-2",
        "S3_PROTO_PREFIX": "proto/",
        "PROTO_LOCAL_DIR": "./tests/fixtures/proto",
        "PROTO_GRPC_OUTPUT_DIR": "./tests/fixtures/grpc"
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_proto_manager(mock_environment_variables, mock_s3_service):
    """ Fixture to create a mock ProtoManager instance. """
    with patch('fp_orchestrator_utils.proto_manager.proto_manager.S3Service', return_value=mock_s3_service):
        manager = ProtoManager()
        return manager

@pytest.fixture
def clean_test_dirs():
    """ Fixture to ensure test directories are clean before each test. """
    proto_dir = Path("./tests/fixtures/proto")
    grpc_output_dir = Path("./tests/fixtures/grpc")

    # Clean up directories if they exist
    if proto_dir.exists():
        shutil.rmtree(proto_dir)
    if grpc_output_dir.exists():
        shutil.rmtree(grpc_output_dir)
    
    # Create directories
    proto_dir.mkdir(parents=True, exist_ok=True)
    grpc_output_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    shutil.rmtree(proto_dir, ignore_errors=True)
    shutil.rmtree(grpc_output_dir, ignore_errors=True)
