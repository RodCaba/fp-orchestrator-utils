import logging
import grpc
from pathlib import Path
import os
import sys
import logging
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

grpc_dir = Path(__file__).parent.parent / "src/grpc"
# Add the grpc directory to sys.path temporarily
grpc_path_str = str(grpc_dir.absolute())
if grpc_path_str not in sys.path:
    sys.path.append(grpc_path_str)
try:
    import orchestrator_service_pb2  # type: ignore
    import orchestrator_service_pb2_grpc  # type: ignore
    import imu_service_pb2  # type: ignore
    import imu_service_pb2_grpc  # type: ignore
except ImportError as e:
    logger.error(f"Failed to import gRPC modules: {e}")
    raise RuntimeError("gRPC modules could not be loaded. Ensure they are generated correctly.")


class OrchestratorClient:
    """
    gRPC client for the orchestrator service.
    """
    def __init__(
            self,
            server_address: str = None,
            timeout: int = 30,
    ):
        """
        Initializes the OrchestratorClient.
        :param server_address: gRPC server address.
        :param timeout: Timeout for gRPC calls in seconds.
        """
        if server_address is None:
            server_address = os.getenv("ORCHESTRATOR_SERVER_ADDRESS", "localhost:50051")
        
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.logger = logging.getLogger(__name__)
        
        self._connect()

    def _connect(self):
        """
        Connects to the gRPC server.
        """
        try:
            self.logger.info(f"Connecting to gRPC server at {self.server_address} with timeout {self.timeout} seconds.")
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = orchestrator_service_pb2_grpc.OrchestratorServiceStub(self.channel)
            self.logger.info(f"Connected to gRPC server at {self.server_address}")
        except Exception as e:
            self.logger.error(f"Failed to connect to gRPC server: {e}")
            raise
        
    def health_check(self):
        """
        Performs a health check on the gRPC server.
        :return: True if the server is healthy, False otherwise.
        """
        try:
            request = orchestrator_service_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(
                request,
                timeout=self.timeout
            )
            self.logger.info(f"Health check response: {response}")
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Health check failed: {e}")
            return False
        
    def _parsed_response(self, response: Any, field_mappings: list[str]) -> dict:
        """
        Parses the gRPC response into a dictionary.
        :param response: gRPC response object.
        :return: Parsed response as a dictionary.
        """
        try:
            status_dict = {}
            for field in field_mappings:
                if hasattr(response, field):
                    status_dict[field] = getattr(response, field)
                else:
                    self.logger.warning(f"Field {field} not found in response.")
            return status_dict
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            return {}
        
    def get_orchestrator_status(self):
        """
        Retrieves the status of the orchestrator.
        :return: Orchestrator status response.
        """
        try:
            request = orchestrator_service_pb2.OrchestratorStatusRequest()
            response = self.stub.OrchestratorStatus(request, timeout=self.timeout)

            return self._parsed_response(
                response,
                field_mappings=["is_ready", "current_activity"]
            )
        except grpc.RpcError as e:
            self.logger.error(f"Failed to get orchestrator status: {e}")
            raise
        
    def send_imu_data(self, device_id: str, imu_data: Any):
        """
        Sends IMU data to the orchestrator.
        :param imu_data: IMU data to send.
        :return: Response from the orchestrator.
        """
        try:
            request = imu_service_pb2.IMUPayload(
                device_id=device_id,
                timestamp=int(datetime.now().timestamp()),
                data=imu_data,
            )
            response = self.stub.ReceiveIMUData(request, timeout=self.timeout)
            return response
        except grpc.RpcError as e:
            self.logger.error(f"Failed to send IMU data: {e}")
            raise
