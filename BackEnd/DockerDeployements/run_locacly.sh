#!/bin/bash

# Settings
IMAGE_NAME="genescope_marimo-fastapi-app"
CONTAINER_NAME="genescope_marimo-fastapi-container"
DOCKERFILE_PATH="DockerDeployements/Dockerfile"


echo "Running new container..."
docker run -d --name $CONTAINER_NAME -p 8000:8000 $IMAGE_NAME

echo "App deployed! Visit: http://0.0.0.0:8000"