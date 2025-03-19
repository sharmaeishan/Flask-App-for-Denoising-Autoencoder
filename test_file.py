import requests
import time
import os

# Flask server URL
url = "http://127.0.0.1:5000/upload"
file_path = "punk_noisyaudio.wav"

# Check if Flask is running before making a request
def is_server_running():
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Wait for Flask to start (if needed)
if not is_server_running():
    print("Waiting for Flask server to start...")
    time.sleep(5)  # Wait 5 seconds before retrying

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    exit(1)

# Upload the file
try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, timeout=15)  # 15s timeout
        response.raise_for_status()  # Raise error if response is not 200 OK

    # Save the denoised output if request was successful
    if response.status_code == 200:
        with open("denoised.wav", "wb") as f:
            f.write(response.content)
        print("✅ Denoised file saved as 'denoised.wav'.")
    else:
        print(f"❌ Server error: {response.status_code} - {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect to Flask server. Make sure it's running.")

except requests.exceptions.Timeout:
    print("❌ Error: Request timed out. Try increasing the timeout value.")

except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
