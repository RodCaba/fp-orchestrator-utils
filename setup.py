from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    requirements = req_file.read().splitlines()

with open("LICENSE", "r") as license_file:
    license_text = license_file.read()

setup(
    name="fp-orchestrator-utils",
    version="0.1.0",
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