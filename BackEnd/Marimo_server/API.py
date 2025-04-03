from typing import Annotated, Callable, Coroutine
from fastapi.responses import HTMLResponse, RedirectResponse
import marimo
from fastapi import FastAPI, Form, Request, Response


# Create a marimo asgi app
server = (
    marimo.create_asgi_app()
    .with_app(path="/eda", root="EDA.py")
    .with_app(path="/major_findings", root="major_findings.py")
)

# Create a FastAPI app
app = FastAPI()

# app.add_middleware(auth_middleware)
# app.add_route("/login", my_login_route, methods=["POST"])

app.mount("/", server.build())

# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)