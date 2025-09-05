# API Documentation for Virtue Multimodal Model

## Overview
This document describes the REST API endpoints for interacting with Virtue, the 270M-parameter multimodal language model. The API is built using FastAPI and supports text-only, image-only, and multimodal inference.

---

## Base URL
```
http://<host>:<port>
```
- Default host: `0.0.0.0`
- Default port: `8000`

---

## API Endpoints

### 1. /generate

**Description:** Generate a text response for a given prompt and optional image.

**Method:** POST

**Request:**
```json
{
  "prompt": "Describe this image.",
  "image_base64": "<base64-encoded image>"
}
```
- `prompt` (string, required): Input text prompt.
- `image_base64` (string, optional): Base64-encoded image data for multimodal generation.

**Response:**
```json
{
  "response": "A beautiful sunrise over mountains..."
}
```
- `response` (string): Generated text.

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt":"What is in this picture?",
        "image_base64":"$(base64 image.jpg)"
      }'
```

---

### 2. /health

**Description:** Health check endpoint to verify API status.

**Method:** GET

**Response:**
```json
{
  "status": "ok"
}
```

---

### 3. /version

**Description:** Retrieve model and API version information.

**Method:** GET

**Response:**
```json
{
  "model_version": "0.1.0",
  "api_version": "1.0.0"
}
```

---

## Authentication (Optional)
- By default, the API is open. For production, integrate API key or OAuth2.
- Example header:
  - `Authorization: Bearer <API_KEY>`

---

## Error Handling
- **400 Bad Request:** Missing or invalid parameters.
- **500 Internal Server Error:** Model crash or server error.

**Error Format:**
```json
{
  "detail": "Error message"
}
```

---

## Rate Limiting
- Implement rate limiting per IP in production (e.g., 60 requests/minute)
- Use FastAPI middleware or reverse proxy.

---

## Deployment Notes
- Configure `uvicorn` with multiple workers: `uvicorn api_server:app --workers 4`
- Use Docker container `virtue-inference` from `docker-compose.yml`.

---

## Example Python Client
```python
import requests
import base64

API_URL = "http://localhost:8000/generate"

with open("image.jpg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode()

payload = {
    "prompt": "What do you see?",
    "image_base64": img_b64
}

resp = requests.post(API_URL, json=payload)
print(resp.json()["response"])
```
