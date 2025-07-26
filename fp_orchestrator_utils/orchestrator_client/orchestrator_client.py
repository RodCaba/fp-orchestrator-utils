import logging
import grpc
from ..src.grpc import orchestrator_service_pb2_grpc, orchestrator_service_pb2

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
            response = self.stub.HealthCheck(grpc.Empty(), timeout=self.timeout)
            self.logger.info(f"Health check response: {response}")
            return response.status == "SERVING"
        except grpc.RpcError as e:
            self.logger.error(f"Health check failed: {e}")
            return False
        
    def get_orchestrator_status(self):
        """
        Retrieves the status of the orchestrator.
        :return: Orchestrator status response.
        """
        try:
            request = orchestrator_service_pb2.OrchestratorStatusRequest()
            response = self.stub.OrchestratorStatus(request, timeout=self.timeout)
            self.logger.info(f"Orchestrator status response: {response}")
            return response
        except grpc.RpcError as e:
            self.logger.error(f"Failed to get orchestrator status: {e}")
            raise