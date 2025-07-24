from setuptools import setup, find_packages

setup(
    name="fp-orchestrator-utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.73.1",
        "grpcio-tools>=1.73.1", 
        "protobuf>=6.31.1",
        "boto3>=1.39.10",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "fp-orchestrator-utils=fp_orchestrator_utils.cli.main:main",
        ],
    },
)