# Variable definitions for sensitive data.
variable "gemini_api_key" {
  description = "API key for Gemini"
  type        = string
  sensitive   = true
}

variable "billing_account" {
  description = "Billing account ID"
  type        = string
}

variable "org_id" {
  description = "Organization ID"
  type        = string
}

variable "gcp_project_name" {
  description = "Project Name"
  type        = string
}

variable "project_number" {
  type = string
}

variable "github_user" {
  description = "GitHub username for the service ID"
  type        = string
}