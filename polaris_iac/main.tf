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
      export TELEGRAM_BOT_PORT="${var.telegram_bot_port}" && export NGROK_URL="${var.ngrok_url}" && export NGROK_PORT="${var.ngrok_port}"
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
    "TELEGRAM_TOKEN=${var.telegram_token}",
    "NGROK_URL=${var.ngrok_url}",
    "NGROK_PORT=${var.ngrok_port}"
  ]

  networks_advanced {
    name = docker_network.polaris_network.name
  }
}
