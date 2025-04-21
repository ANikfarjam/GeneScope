from fastapi import FastAPI
import marimo
from fastapi.responses import RedirectResponse

# Create a marimo multi-app server
server = (
    marimo.create_asgi_app()
    .with_app(path="/", root="app.py")  # This sets the homepage
    .with_app(path="/eda", root="EDA.py")
    .with_app(path="/major_findings", root="major_findings.py")
    .with_app(path="/ml", root="MLs.py")
)

# Create the FastAPI app and mount the Marimo server
app = FastAPI()

# Mount the multi-app Marimo server at root
app.mount("/", server.build())

# Run it
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
