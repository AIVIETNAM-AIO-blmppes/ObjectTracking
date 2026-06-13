# import requests
# import json

# url = "http://localhost:8000/detect"
# image_path = "test_image.jpg"

# with open(image_path, "rb") as image_file:
#     files = {"file": (image_path, image_file, "image/jpeg")}
#     print(f"Sending {image_path} to {url}...")

#     response = requests.post(url, files=files)

# if response.status_code == 200:
#     # Print the JSON nicely formatted
#     print("Success!")
#     print(json.dumps(response.json(), indent=2))
# else:
#     print(f"Error {response.status_code}: {response.text}")

import requests

url = "http://localhost:8000/track-video"
video_path = "C:/Users/ADMIN/Documents/ObjectTracking/08fd33_4_short.mp4"

print(f"Sending video to pipeline.")

with open(video_path, "rb") as f:
    files = {"file": ("test_video.mp4", f, "video/mp4")}
    response = requests.post(url, files=files, timeout=None)

if response.status_code == 200:
    print("Success! Output saved to:")
    print(response.json()["output_video_path"])
else:
    print(f"Error {response.status_code}:", response.text)