import os
from apscheduler.schedulers.blocking import BlockingScheduler
from google_drive_utils import process_pdfs_from_drive
from dotenv import load_dotenv

load_dotenv()

def run_process_pdfs():
    print("Running process_pdfs_from_drive...")
    process_pdfs_from_drive()
    print("Finished running process_pdfs_from_drive.")

if __name__ == "__main__":
    scheduler = BlockingScheduler()

    # Schedule the job to run every day at 3:00 AM
    scheduler.add_job(run_process_pdfs, 'cron', hour=3, minute=0)

    print("Scheduler starting...")
    scheduler.start()