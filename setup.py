"""
Setup script for PyExplore package.
"""

from setuptools import setup, find_packages

setup(
    name="pyexplore",
    version="0.1.0",
    description="Exploration Strategies for Deep Reinforcement Learning",
    author="Yahia Shaaban",
    author_email="Yahia.shaaban@mbzuai.ac.ae",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "gymnasium>=0.26.0",
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 