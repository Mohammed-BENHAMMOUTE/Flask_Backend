from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Path to your service account key file
SERVICE_ACCOUNT_FILE = r'C:\Users\ok\Downloads\chatbot-esj-3246e2b7aea5.json'

# Define the required scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Create credentials using the service account
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Build the Drive service
service = build('drive', 'v3', credentials=credentials)

# Function to list files in a specific folder
def list_files_in_folder(folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, spaces='drive',
                                   fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")

# Function to download a file
def download_file(file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    print(f"Downloaded {file_name}.")

if __name__ == '__main__':
    # Replace with your folder ID
    FOLDER_ID = '1op8LeB2YWObntdphtHibTyCwWsuYiijU'
    list_files_in_folder(FOLDER_ID)
    
    # Example: Download the first file
    files = service.files().list(q=f"'{FOLDER_ID}' in parents", spaces='drive',
                                 fields="files(id, name)").execute().get('files', [])
    if files:
        download_file(files[0]['id'], files[0]['name'])
