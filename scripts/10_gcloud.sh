#!/usr/bin/env bash
# Install Google Cloud CLI (gcloud + gsutil) for downloading Waymo Open Dataset tfrecords.
set -eo pipefail
if ! command -v gcloud >/dev/null; then
  apt-get update -y && apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -y && apt-get install -y google-cloud-cli
fi
gcloud --version | head -1
