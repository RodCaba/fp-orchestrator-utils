from setuptools import setup, find_packages
import os
from pathlib import Path

here = Path(__file__).parent.resolve()

with open(here / "README.md", "r") as fh:
    long_description = fh.read()

requirements_file = here / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as req_file:
        requirements = [
            line.strip() for line in req_file.read().splitlines() if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "grpcio>=1.73.1",
        "grpcio-tools>=1.73.1", 
        "protobuf>=6.31.1",
        "boto3>=1.39.10",
        "python-dotenv>=1.0.0"
    ]


with open(here / "LICENSE", "r") as license_file:
    license_text = license_file.read()

setup(
    name="fp-orchestrator-utils",
    version="0.1.18",
    author="Rodrigo",
    author_email="rodser4@gmail.com",
    description="Utilities for the FP Orchestrator, including CLI tools for managing Protocol Buffers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RodCaba/fp-orchestrator-utils",
    license=license_text,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries"
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fp-orchestrator-utils=fp_orchestrator_utils.cli.main:main",
        ],
    },
)