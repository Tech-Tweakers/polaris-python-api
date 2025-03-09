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
}

resource "docker_volume" "mongodb_data" {
  name = "mongodb_data"
}

resource "null_resource" "deploy" {
  provisioner "local-exec" {
    command = <<EOT
      docker-compose up -d
      ./scripts/setup_ngrok.sh
    EOT
  }
}
