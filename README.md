# FP Orchestrator Utils
This package provides utilities for managing Protocol Buffers and S3 storage integration in the sensor layer of my Final Project.

## Features

- **Protocol Buffer Management**: Simplifies the creation and management of Protocol Buffers for gRPC services. This protocol buffers are stored in S3 for centralized access.
- **S3 Storage Integration**: Provides a service for uploading, downloading, and managing S3 objects.
- **CLI Commands**: Includes command-line interface commands for managing Protocol Buffers and S3 storage.

## Installation
To install the package, run:
```bash
pip install fp-orchestrator-utils
```

## Usage

Setup your environment by creating a `.env` file with the environment variables required for S3 and Protocol Buffers:
```plaintext
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_s3_bucket_name
S3_PROTO_PREFIX=your_s3_proto_prefix
```

### Command Line Interface
You can use the CLI to manage Protocol Buffers and S3 storage. The available commands are:
```bash
# Download Protocol Buffers from S3, it will download all the proto files from the defined S3 bucket prefix.
fp-orchestrator-utils proto download
# Generate Python code from the downloaded Protocol Buffers
fp-orchestrator-utils proto generate
# Upload Protocol Buffers to S3
fp-orchestrator-utils proto upload
```
