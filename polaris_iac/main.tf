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

resource "null_resource" "deploy_polaris" {
  provisioner "local-exec" {
    command = <<EOT
      echo "🔥 Subindo Polaris no laptop..."
      export $(grep -v '^#' .env | xargs)
      docker-compose --env-file .env up -d
      ./scripts/setup_ngrok.sh
    EOT
  }
}
