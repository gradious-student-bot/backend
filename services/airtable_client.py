import logging
from pyairtable import Api
import config

logger = logging.getLogger(__name__)

_api = Api(config.AIRTABLE_TOKEN)

lead_table = _api.table(config.AIRTABLE_BASE_ID, config.AIRTABLE_LEAD_TABLE)
metrics_table = _api.table(config.AIRTABLE_BASE_ID, config.AIRTABLE_METRICS_TABLE)

logger.info("Airtable client initialized")
logger.info(f"Lead table: {config.AIRTABLE_LEAD_TABLE}")
logger.info(f"Metrics table: {config.AIRTABLE_METRICS_TABLE}")


def _clean_data(data: dict) -> dict:
    """
    Remove None values before writing to Airtable.
    Keeps False values because checkbox fields need False.
    """
    return {k: v for k, v in data.items() if v is not None}


def write_lead_table(lead_data: dict) -> dict:
    """
    Writes lead profile details to Lead_Table.
    """
    clean_data = _clean_data(lead_data)

    logger.info(f"Writing Lead_Table record: {clean_data.get('lead_id')}")
    logger.info(f"Lead_Table fields: {list(clean_data.keys())}")

    try:
        record = lead_table.create(clean_data)
        logger.info(f"Lead_Table record written: {record.get('id')}")
        return record
    except Exception as e:
        logger.error(f"Failed to write Lead_Table record: {str(e)}", exc_info=True)
        raise


def write_lead_metrics(metrics_data: dict) -> dict:
    """
    Writes call metrics details to Lead_metrics.
    """
    clean_data = _clean_data(metrics_data)

    logger.info(f"Writing Lead_metrics record: {clean_data.get('conversation_id')}")
    logger.info(f"Lead_metrics fields: {list(clean_data.keys())}")

    try:
        record = metrics_table.create(clean_data)
        logger.info(f"Lead_metrics record written: {record.get('id')}")
        return record
    except Exception as e:
        logger.error(f"Failed to write Lead_metrics record: {str(e)}", exc_info=True)
        raise