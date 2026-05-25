import os
import subprocess
import sys

def download_google_drive_folder():
    # Folder URL
    folder_url = "https://drive.google.com/drive/folders/1_X7A5IrsR2qyAnyyM-gTS0c5nelWN-q5"
    
    # Path to Downloads folder
    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Drive_Data')
    
    os.makedirs(downloads_path, exist_ok=True)
    
    print(f"Data will be saved to: {downloads_path}")
    print("Installing 'gdown' if not already installed...")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
    
    print("Starting download from Google Drive...")
    
    # Set environment variables to support UTF-8 characters in folder/file names
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    try:
        subprocess.run([sys.executable, "-m", "gdown", "--folder", folder_url, "-O", downloads_path], env=env, check=True)
        print("\nDownload complete! You can find the files at: " + downloads_path)
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during download: {e}")

if __name__ == "__main__":
    download_google_drive_folder()
