#!/bin/bash

# Ensure script exits if a command fails
set -e

# Variables
docker_manage_script="scripts/docker_manage.sh"

# Function to show usage
usage() {
  echo "Usage: $0 {run|deploy|build|up|down|buildc|upc|downc|help} [arguments]"
  echo "Commands:"
  echo "  run        Run the specified script from the scripts directory."
  echo "  deploy     Deploy the application using Docker."
  echo "  build      Build the Docker image."
  echo "  up         Run the Docker container."
  echo "  down       Stop and remove the Docker container."
  echo "  buildc     Build the Docker Compose services."
  echo "  upc        Start the Docker Compose services."
  echo "  downc      Stop and remove the Docker Compose services."
  echo "  help       Display this help message."
  echo
  echo "Examples:"
  echo "  $0 run my_script.sh"
  echo "  $0 deploy"
  echo "  $0 build"
  echo "  $0 up"
  echo "  $0 down"
  echo "  $0 buildc"
  echo "  $0 upc"
  echo "  $0 downc"
}

# Check if at least one argument is passed
if [ $# -lt 1 ]; then
  usage
  exit 1
fi

COMMAND=$1
shift

case $COMMAND in
  run)
    if [ $# -lt 1 ]; then
      echo "Error: You must specify the script to run."
      usage
      exit 1
    fi
    SCRIPT_NAME=$1
    shift
    if [ -f "scripts/$SCRIPT_NAME" ]; then
      echo "Running script: scripts/$SCRIPT_NAME"
      bash "scripts/$SCRIPT_NAME" "$@"
    else
      echo "Error: Script scripts/$SCRIPT_NAME does not exist."
      exit 1
    fi
    ;;
  deploy)
    if [ -f "scripts/deploy.sh" ]; then
      echo "Deploying application..."
      bash "scripts/deploy.sh" "$@"
    else
      echo "Error: Deploy script scripts/deploy.sh does not exist."
      exit 1
    fi
    ;;
  build)
    echo "Building Docker image..."
    docker build -t exoplanet_ml-web:latest .
    echo "Docker image built successfully!"
    ;;
  up)
    echo "Running Docker container..."
    docker run -d --name exoplanet_ml_container -p 8501:8501 exoplanet_ml-web:latest
    # docker-compose up
    echo "Docker container is running!"
    ;;
  down)
    echo "Stopping and removing Docker container..."
    docker stop exoplanet_ml_container
    docker rm exoplanet_ml_container
    echo "Docker container stopped and removed!"
    ;;
  buildc)
    echo "Building Docker Compose services..."
    bash "$docker_manage_script" build
    ;;
  upc)
    echo "Starting Docker Compose services..."
    bash "$docker_manage_script" up
    ;;
  downc)
    echo "Stopping and removing Docker Compose services..."
    bash "$docker_manage_script" down
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
