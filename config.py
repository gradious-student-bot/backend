import logging
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    AIRTABLE_TOKEN: str
    AIRTABLE_BASE_ID: str
    AIRTABLE_TABLE_NAME: str = "Leads"

    class Config:
        env_file = ".env"


settings = Settings()
logger.info("Settings loaded successfully")