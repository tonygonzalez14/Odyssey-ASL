# This script organizes the ASL videos into folders based on the word ("gloss") being signed.
# It reads video metadata from the WLASL JSON file, groups videos by gloss, and copies them into corresponding folders.
import os
import json
import shutil

# Load the JSON file
# WLASL_v0.3 was modified to WLASL_v0.4 to only contain three words
with open('WLASL_v0.4.json', 'r') as f:
    video_data = json.load(f)

# Paths to the folders where all videos are currently located
videos_folder = 'videos'
missing_videos_folder = 'missing_videos'

# Path to the destination directory where the word-named folders will be created
destination_root_folder = 'grouped_videos_book_drink_computer2'

# Ensure the destination root folder exists
if not os.path.exists(destination_root_folder):
    os.makedirs(destination_root_folder)

# Process each gloss entry
video_counter = 0
for entry in video_data:
    gloss = entry["gloss"]
    
    # Create the destination folder for the gloss if it doesn't exist
    destination_folder = os.path.join(destination_root_folder, gloss)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Process each instance within the gloss
    for instance in entry["instances"]:
        video_id = instance["video_id"]
        
        # Construct the video file path
        video_file = os.path.join(videos_folder, f'{video_id}.mp4')  # Assuming videos are all .mp4 files
        missing_video_file = os.path.join(missing_videos_folder, f'{video_id}.mp4')

        # Check if the video exists and copy it to the appropriate folder
        if os.path.exists(video_file):
            destination_file = os.path.join(destination_folder, f'{video_id}.mp4')
            shutil.copy(video_file, destination_file)
            print(f'Copied {video_id}.mp4 to {destination_folder}')
        elif os.path.exists(missing_video_file): # Check if video not found in "videos" checking missing_videos and copy it to the appropriate folder
            continue
            destination_file = os.path.join(destination_folder, f'{video_id}.mp4')
            shutil.copy(missing_video_file, destination_file)
            print(f'Copied {video_id}.mp4 to {destination_folder}')
        else: # Video not found in either folder
            print(f'Video {video_id}.mp4 not found in folders')
        video_counter += 1

print(f'{video_counter} Videos Processed')
