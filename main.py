import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from api.routes import router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gradious Lead Agent",
    description="Stateful voice agent for lead conversion counseling",
    version="1.0.0",
)

logger.info("FastAPI app initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware added")

app.include_router(router)


@app.get("/health")
async def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

# Load the templates/chat.html file and serve it at the /chat endpoint
@app.get("/chat", response_class=HTMLResponse)
async def chat():
    logger.info("Serving chat interface")
    with open("templates/chat.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run(
        "main:app",
        # host="0.0.0.0",
        port=8000,
        log_level="info",
        # reload=True
    )