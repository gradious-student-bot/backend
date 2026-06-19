import logging
from pyairtable import Api
from config import settings

logger = logging.getLogger(__name__)

_api = Api(settings.AIRTABLE_TOKEN)
_table = _api.table(settings.AIRTABLE_BASE_ID, settings.AIRTABLE_TABLE_NAME)

logger.info(f"Airtable client initialized - Table: {settings.AIRTABLE_TABLE_NAME}")


def write_lead(lead_data: dict) -> dict:
    """
    Creates or updates a lead record in Airtable.
    Filters out None values before writing.
    """
    clean_data = {k: v for k, v in lead_data.items() if v is not None}
    logger.info(f"Writing lead record to Airtable: {clean_data.get('Lead ID')}")
    logger.debug(f"Lead data fields: {list(clean_data.keys())}")
    
    try:
        record = _table.create(clean_data)
        logger.info(f"Lead record successfully written with ID: {record.get('id')}")
        return record
    except Exception as e:
        logger.error(f"Failed to write lead record: {str(e)}", exc_info=True)
        raise