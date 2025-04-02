#!/usr/bin/env bash

gcloud projects create $GCP_PROJECT_NAME
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_NAME" --format="value(projectNumber)")

# you need to link a project to a billing account to be able to use it
gcloud projects add-iam-policy-binding $GCP_PROJECT_NAME --member="user:oliver@ishango.ai" --role="roles/owner"
read -p "Waiting for billing account to be enabled... Continue?"
# gcloud billing projects link $GCP_PROJECT_NAME --billing-account=01AC44-DF4B27-C9AF71

gcloud storage buckets create gs://${GCP_PROJECT_NAME}-state-bucket \
    --project=$GCP_PROJECT_NAME \
    --default-storage-class=STANDARD \
    --location=EUROPE-WEST2 \
    --uniform-bucket-level-access \
    --public-access-prevention 


gcloud iam service-accounts create tf-provision \
    --project=$GCP_PROJECT_NAME \
    --description="SA to provision infrastructure with Code (IaC) using tofu " \
    --display-name="Tofu Provisioner"

gcloud projects add-iam-policy-binding $GCP_PROJECT_NAME \
    --member="serviceAccount:tf-provision@$GCP_PROJECT_NAME.iam.gserviceaccount.com" \
    --role="roles/owner"

# generate a key for the service account to be used by tofu locally
gcloud iam service-accounts keys create tofu-key.json \
    --iam-account=tf-provision@$GCP_PROJECT_NAME.iam.gserviceaccount.com \
    --key-file-type=json

# export GOOGLE_CREDENTIALS="$(cat tofu-key.json)"
# echo $GOOGLE_CREDENTIALS

# workload identity - github actions will be able to authenticate as the service account without needing to store the key
MAX_RETRIES=10
RETRY_DELAY=10

for i in $(seq 1 $MAX_RETRIES); do
  gcloud iam workload-identity-pools create github \
    --project=$GCP_PROJECT_NAME \
    --location="global" \
    --description="GitHub pool" \
    --display-name="GitHub pool" && break

  echo "Attempt $i failed. Retrying in $RETRY_DELAY seconds..."
  sleep $RETRY_DELAY
done

if [[ $i -gt $MAX_RETRIES ]]; then
  echo "ERROR: Failed to create workload identity pool after $MAX_RETRIES attempts."
  exit 1
fi

gcloud iam workload-identity-pools providers create-oidc "github" \
  --project=$GCP_PROJECT_NAME \
  --location="global" \
  --workload-identity-pool="github" \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.event_name=assertion.event_name" \
  --attribute-condition="attribute.event_name == 'push' || attribute.event_name == 'pull_request'" \
  --issuer-uri="https://token.actions.githubusercontent.com"


gcloud iam service-accounts add-iam-policy-binding "tf-provision@${GCP_PROJECT_NAME}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_NAME}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.event_name/push"

gcloud iam service-accounts add-iam-policy-binding "tf-provision@${GCP_PROJECT_NAME}.iam.gserviceaccount.com" \
  --project="${GCP_PROJECT_NAME}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.event_name/pull_request"

export GOOGLE_CREDENTIALS="$(cat tofu-key.json)"
echo $GOOGLE_CREDENTIALS

# tofu init -backend-config="bucket=${GCP_PROJECT_NAME}-state-bucket"

gcloud services enable iamcredentials.googleapis.com --project=$GCP_PROJECT_NAME
gcloud services enable artifactregistry.googleapis.com --project=$GCP_PROJECT_NAME
gcloud services enable iam.googleapis.com --project=$GCP_PROJECT_NAME
gcloud services enable run.googleapis.com --project=$GCP_PROJECT_NAME