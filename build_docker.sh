#!/bin/bash
set -e
IMAGE_NAME="superneed/lottery:latest"
echo "Target: $IMAGE_NAME"
if ! docker buildx inspect mybuilder > /dev/null 2>&1; then
    docker buildx create --use --name mybuilder
else
    docker buildx use mybuilder
fi
docker buildx build --platform linux/amd64,linux/arm64 -t "$IMAGE_NAME" --push .
