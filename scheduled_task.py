import schedule
import time
from google_drive_utils import process_pdfs_from_drive
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def job():
    logger.info("Running PDF processing job from Google Drive...")
    process_pdfs_from_drive()
    logger.info("PDF processing job from Google Drive completed.")


# Schedule the job to run every day at 3 AM
schedule.every().day.at("23:47").do(job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait for one minute
