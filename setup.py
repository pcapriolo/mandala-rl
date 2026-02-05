"""Setup script for mandala-rl package."""
from setuptools import setup, find_packages

setup(
    name="mandala-rl",
    version="0.1.0",
    description="AlphaZero-style RL for Mandala card game",
    author="Paul Capriolo",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "flyte": [
            "flytekit>=1.10.0",
        ]
    },
    python_requires=">=3.9",
)
