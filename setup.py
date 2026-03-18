from setuptools import setup, find_packages

setup(
    name        = "video-tracking-analytics",
    version     = "1.0.0",
    description = "Real-time multi-object tracking and analytics pipeline (YOLOv8 + ByteTrack)",
    author      = "Tajwar",
    packages    = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ultralytics>=8.0.0",
        "boxmot>=10.0.0",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)