# Fake News Detector

A machine learning-powered **Fake News Detection** system built with **FastAPI**, **Docker**, and **Scikit-learn**.

---

## Features
- Detects fake news using ML models (Logistic Regression, Random Forest, etc.)
- RESTful API with FastAPI
- Dockerized for easy deployment
- Swagger UI for API testing

## API Endpoints

| Method | Endpoint    | Description           |
|--------|-------------|-----------------------|
| GET    | `/`         | Health check          |
| POST   | `/predict/` | Fake/Real prediction  |
| POST   | `/set_model/` | Switch ML model     |

## Example Server Logs

These are example server logs when running the FastAPI app with Uvicorn:

```bash
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.17.0.1:55626 - "GET /docs HTTP/1.1" 200 OK
INFO:     172.17.0.1:55626 - "GET /openapi.json HTTP/1.1" 200 OK
INFO:     172.17.0.1:55628 - "GET / HTTP/1.1" 200 OK
INFO:     172.17.0.1:56382 - "GET / HTTP/1.1" 200 OK
INFO:     172.17.0.1:64282 - "POST /predict/ HTTP/1.1" 200 OK
INFO:     172.17.0.1:59828 - "POST /predict/ HTTP/1.1" 422 Unprocessable Entity
INFO:     172.17.0.1:60682 - "POST /set_model/?model_name=RandomForest HTTP/1.1" 200 OK

