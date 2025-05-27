# Vertex Module

CLI for any model hosted on **Vertex AI** (Gemini, Imagen, Claude, etc.).

## Authentication

Vertex uses **Google Cloud IAM**, not an API key.

```bash
# local workstation
gcloud auth application-default login

# CI / servers
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
