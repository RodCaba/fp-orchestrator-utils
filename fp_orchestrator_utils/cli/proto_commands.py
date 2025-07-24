import logging
from ..proto_manager import ProtoManager

logger = logging.getLogger(__name__)


def setup_proto_commands(subparsers):
    """Setup proto-related CLI commands."""
    proto_parser = subparsers.add_parser('proto', help='Protocol Buffer utilities')
    proto_subparsers = proto_parser.add_subparsers(dest='proto_command', required=True)
    
    # Download command
    download_parser = proto_subparsers.add_parser('download', help='Download .proto files from S3')
    download_parser.set_defaults(func=cmd_download_protos)
    
    # Upload command
    upload_parser = proto_subparsers.add_parser('upload', help='Upload .proto files to S3')
    upload_parser.set_defaults(func=cmd_upload_protos)
    
    # Generate command
    generate_parser = proto_subparsers.add_parser('generate', help='Generate gRPC code from .proto files')
    generate_parser.set_defaults(func=cmd_generate_grpc)
    
    # Check command
    check_parser = proto_subparsers.add_parser('check', help='Check S3 connection')
    check_parser.set_defaults(func=cmd_check_connection)


def cmd_download_protos(args):
    """Download proto files command."""
    try:
        manager = ProtoManager()
        if not manager.check_connection():
            logger.error("Failed to connect to S3")
            return 1
            
        files = manager.download_protos()
        logger.info(f"Successfully downloaded {len(files)} files")
        return 0
    except Exception as e:
        logger.error(f"Error downloading protos: {e}")
        return 1


def cmd_upload_protos(args):
    """Upload proto files command."""
    try:
        manager = ProtoManager()
        if not manager.check_connection():
            logger.error("Failed to connect to S3")
            return 1
            
        manager.upload_protos()
        logger.info("Successfully uploaded proto files")
        return 0
    except Exception as e:
        logger.error(f"Error uploading protos: {e}")
        return 1


def cmd_generate_grpc(args):
    """Generate gRPC code command."""
    try:
        manager = ProtoManager()
        files = manager.generate_grpc_code()
        logger.info(f"Successfully generated gRPC code for {len(files)} files")
        return 0
    except Exception as e:
        logger.error(f"Error generating gRPC code: {e}")
        return 1


def cmd_check_connection(args):
    """Check S3 connection command."""
    try:
        manager = ProtoManager()
        if manager.check_connection():
            logger.info("Successfully connected to S3")
            return 0
        else:
            logger.error("Failed to connect to S3")
            return 1
    except Exception as e:
        logger.error(f"Error checking connection: {e}")
        return 1