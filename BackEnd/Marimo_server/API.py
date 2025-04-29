from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Create the FastAPI app
app = FastAPI()

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount the Marimo server
app.mount("/", server.build())

# Run it
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
