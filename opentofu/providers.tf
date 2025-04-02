terraform {
  backend "gcs" {
    prefix = "tofu"
  }
}

# Provider configuration (optional, if not using Application Default Credentials)
provider "google" {
  project = var.gcp_project_name
}