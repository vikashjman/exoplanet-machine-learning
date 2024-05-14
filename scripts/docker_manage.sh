#!/bin/bash

# Ensure script exits if a command fails
set -e

# Variables
docker_compose_file="docker-compose.yml"

# Function to show usage
usage() {
  echo "Usage: $0 {build|up|down|help} [arguments]"
  echo "Commands:"
  echo "  build      Build the Docker Compose services."
  echo "  up         Start the Docker Compose services."
  echo "  down       Stop and remove the Docker Compose services."
  echo "  help       Display this help message."
  echo
  echo "Examples:"
  echo "  $0 build"
  echo "  $0 up"
  echo "  $0 down"
}

# Check if at least one argument is passed
if [ $# -lt 1 ]; then
  usage
  exit 1
fi

COMMAND=$1
shift

case $COMMAND in
  build)
    echo "Building Docker Compose services..."
    docker-compose -f "$docker_compose_file" build
    echo "Docker Compose services built successfully!"
    ;;
  up)
    echo "Starting Docker Compose services..."
    docker-compose -f "$docker_compose_file" up -d
    echo "Docker Compose services are up and running!"
    ;;
  down)
    echo "Stopping and removing Docker Compose services..."
    docker-compose -f "$docker_compose_file" down
    echo "Docker Compose services stopped and removed!"
    ;;
  help)
    usage
    ;;
  *)
    echo "Error: Unknown command: $COMMAND"
    usage
    exit 1
    ;;
esac
