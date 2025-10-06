resource "google_artifact_registry_repository" "artifact_registry_repository" {
  cleanup_policy_dry_run = true
  format                 = "DOCKER"
  location               = "europe-west2"
  mode                   = "STANDARD_REPOSITORY"
  project                = var.gcp_project_name
  repository_id          = "${var.gcp_project_name}-gcr"
}

# Grant Docker Action access to push to the GAR 
resource "google_artifact_registry_repository_iam_member" "workload_identity_binding" {
  project    = google_artifact_registry_repository.artifact_registry_repository.project
  location   = google_artifact_registry_repository.artifact_registry_repository.location
  repository = google_artifact_registry_repository.artifact_registry_repository.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "principalSet://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/github/attribute.event_name/push"
}

resource "google_service_account" "cloud_run_service_account" {
  account_id   = "cloud-run-service"
  display_name = "Cloud Run Service Account"
  project      = var.gcp_project_name
}

resource "google_cloud_run_v2_service" "cloud_run_service" {
  client         = "gcloud"
  client_version = "511.0.0"
  ingress        = "INGRESS_TRAFFIC_ALL"
  launch_stage   = "GA"
  location       = "europe-west2"
  name           = "${var.gcp_project_name}-service"
  project        = var.gcp_project_name
  template {
    containers {
      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
      image = "us-docker.pkg.dev/cloudrun/container/hello"
      name  = "${var.gcp_project_name}-image-1"
      ports {
        container_port = 8080
        name           = "http1"
      }
      resources {
        cpu_idle = true
        limits = {
          cpu    = "1000m"
          memory = "1024Mi"
        }
        startup_cpu_boost = true
      }
      startup_probe {
        failure_threshold     = 1
        initial_delay_seconds = 0
        period_seconds        = 240
        tcp_socket {
          port = 8080
        }
        timeout_seconds = 240
      }
    }
    max_instance_request_concurrency = 80
    scaling {
      max_instance_count = 10
    }
    service_account = google_service_account.cloud_run_service_account.email
    timeout         = "300s"
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# # Define the Service Directory namespace
# resource "google_service_directory_namespace" "service_directory_namespace" {
#   provider     = google-beta
#   namespace_id = "deployed-services"
#   location     = "europe-west2"
#   project      = var.gcp_project_name
# }

# # Define the Service Directory service
# resource "google_service_directory_service" "cloud_run_service" {
#   provider   = google-beta
#   service_id = "${var.gcp_project_name}-service-${var.github_user}"
#   namespace  = google_service_directory_namespace.service_directory_namespace.id
#
#   metadata = {
#     gcr_uri = google_cloud_run_v2_service.cloud_run_service.uri
#     region  = "europe-west2"
#   }
# }