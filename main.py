import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config import settings

logger = logging.getLogger(__name__)

# Set OpenAI key for LangChain
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
logger.info("OpenAI API key configured")

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