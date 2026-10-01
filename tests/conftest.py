import os

# CORS origins are read when app.main is imported, so set a known value before any test imports it.
# Environment variables take precedence over .env in pydantic-settings.
os.environ["CORS_ORIGINS"] = "https://allowed.example, http://localhost:5173/"
