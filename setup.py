"""Setup script for mandala-rl package with C++ MCTS extension."""
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy

setup(
    name="mandala-rl",
    version="0.1.0",
    description="AlphaZero-style RL for Mandala card game",
    author="Paul Capriolo",
    packages=find_packages(),
    ext_modules=[
        Pybind11Extension(
            "mcts_cpp",
            [
                "cpp/bindings.cpp",
                "cpp/mcts_node.cpp",
                "cpp/mandala_game.cpp",
                "cpp/lost_cities_game.cpp",
                "cpp/batched_mcts.cpp",
            ],
            include_dirs=[numpy.get_include()],
            cxx_std=17,
            extra_compile_args=["-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pybind11>=2.11.0",
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
