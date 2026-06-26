import logging
import os
from dotenv import load_dotenv

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration variables from environment

# LLM Model config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_THINK_MODEL = os.environ.get("OPENAI_THINK_MODEL", "gpt-5")
OPENAI_FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-5-mini")

AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_LEAD_TABLE = os.environ.get("AIRTABLE_LEAD_TABLE", "Lead_Table")
AIRTABLE_METRICS_TABLE = os.environ.get("AIRTABLE_METRICS_TABLE", "Lead_metrics")

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


logger.info("Configuration loaded successfully")