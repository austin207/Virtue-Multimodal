# scripts/deployment/docker_build.py

"""
Build Docker images for training and inference.
"""

import os

def build_training():
    os.system("docker build -f docker/Dockerfile.training -t virtue-training .")

def build_inference():
    os.system("docker build -f docker/Dockerfile.inference -t virtue-inference .")

if __name__ == "__main__":
    build_training()
    build_inference()
