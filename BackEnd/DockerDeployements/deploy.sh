#!/bin/bash

# Settings
LOCAL_IMAGE_NAME="genescope_marimo-fastapi-app"
REMOTE_IMAGE_NAME="anikfarjam/genescope_marimo"
CONTAINER_NAME="genescope_marimo-fastapi-container"
DOCKERFILE_PATH="DockerDeployements/Dockerfile"

cd ..

# Check if Marimo_server exists
if [ ! -d "Marimo_server" ]; then
  echo -e "\033[0;31mERROR: Marimo_server directory not found. Please run this script from the project root.\033[0m"
  exit 1
fi

# Remove any existing container
if docker ps -a --format '{{.Names}}' | grep -Eq "^$CONTAINER_NAME\$"; then
  echo -e "\033[1;33mRemoving existing container: $CONTAINER_NAME\033[0m"
  docker rm -f "$CONTAINER_NAME"
fi

# Build Apple M1/M2 native (for local testing)
echo -e "\033[1;34mBuilding local Apple Silicon (arm64) image for local testing...\033[0m"
if docker buildx build --platform linux/arm64/v8 -t $LOCAL_IMAGE_NAME-arm64 -f $DOCKERFILE_PATH .; then
  echo -e "\033[0;32mLocal Apple Silicon image built successfully: $LOCAL_IMAGE_NAME-arm64\033[0m"
else
  echo -e "\033[0;31mFailed to build Apple Silicon version.\033[0m"
  exit 1
fi

# Build Linux amd64 version (for DockerHub and Render)
echo -e "\033[1;34mBuilding linux/amd64 image for Render deployment...\033[0m"
if docker buildx build --platform linux/amd64 -t $LOCAL_IMAGE_NAME -f $DOCKERFILE_PATH .; then
  echo -e "\033[0;32mlinux/amd64 image built successfully: $LOCAL_IMAGE_NAME\033[0m"
else
  echo -e "\033[0;31mFailed to build linux/amd64 version.\033[0m"
  exit 1
fi

# Save Docker image locally
echo -e "\033[1;34mSaving Docker image to local file: $CONTAINER_NAME.tar ...\033[0m"
docker save -o "$CONTAINER_NAME.tar" "$LOCAL_IMAGE_NAME"
echo -e "\033[0;32mImage saved as $CONTAINER_NAME.tar\033[0m"

# Tag the image for Docker Hub (optional backup)
echo -e "\033[1;34mTagging image for Docker Hub (backup and latest)...\033[0m"
docker tag $LOCAL_IMAGE_NAME $REMOTE_IMAGE_NAME:backup-before-latest
docker tag $LOCAL_IMAGE_NAME $REMOTE_IMAGE_NAME:latest

# Login to Docker Hub
echo -e "\033[1;34mLogging into Docker Hub...\033[0m"
docker login || exit 1

# Push to Docker Hub
echo -e "\033[1;34mPushing both backup and latest tags to Docker Hub...\033[0m"
docker push $REMOTE_IMAGE_NAME:backup-before-latest
docker push $REMOTE_IMAGE_NAME:latest
echo -e "\033[0;32m✅ Push complete. Image is live at: https://hub.docker.com/r/anikfarjam/genescope_marimo/tags\033[0m"
