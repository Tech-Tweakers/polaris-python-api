terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_network" "polaris_network" {
  name = "polaris_network"
  check_duplicate = true
}

resource "docker_volume" "mongodb_data" {
  name = "mongodb_data"
}

resource "null_resource" "deploy_ngrok" {
  provisioner "local-exec" {
    command = <<EOT
      echo "🔥 Subindo Ngrok..."
      export MONGO_USER="${var.mongo_user}"
      export MONGO_PASSWORD="${var.mongo_password}"
      export POLARIS_API_PORT="${var.polaris_api_port}"
      export TELEGRAM_BOT_PORT="${var.telegram_bot_port}"
      export TELEGRAM_TOKEN="${var.telegram_token}"
      
      docker-compose up -d
      ./scripts/setup_ngrok.sh
    EOT
  }
}

resource "docker_container" "polaris-api" {
  image    = "polaris-api:latest"
  name     = "polaris-api"
  restart  = "always"

  env = [
    "MONGO_USER=${var.mongo_user}",
    "MONGO_PASSWORD=${var.mongo_password}",
    "POLARIS_API_PORT=${var.polaris_api_port}",
    "TELEGRAM_BOT_PORT=${var.telegram_bot_port}",
    "TELEGRAM_TOKEN=${var.telegram_token}"
  ]

  networks_advanced {
    name = docker_network.polaris_network.name
  }
}
