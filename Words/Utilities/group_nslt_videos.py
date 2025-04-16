# This script organizes the ASL videos into folders based on the word being signed.
# It reads video metadata from the nslt JSON file and uses a word mapping from the class_list text file to identify the signed word.
# For each video, the script creates a folder named after the signed word, and copies the video into that folder.
import os
import json
import shutil

# Load the JSON file
# Change the filename below to extract and group more/less videos
with open('nslt_10.json', 'r') as f:
    video_data = json.load(f)

# Load the mapping file
word_mapping = {}
with open('wlasl_class_list.txt', 'r') as f:
    for line in f:
        word_id, word = line.strip().split('\t')
        word_mapping[int(word_id)] = word  # Convert word_id to integer for correct matching

# Function to get the word being signed in a video
def get_signed_word(action):
    word_id = action[0]  # The first integer represents the word ID
    return word_mapping.get(word_id, "Unknown word")

# Path to the folder where all videos are currently located
videos_folder = 'videos'
missing_videos_folder = 'missing_videos'

# Path to the destination directory where the word-named folders will be created
destination_root_folder = 'grouped_videos'

# Create directories for each word and copy the corresponding video to its folder
video_counter = 0
for video_id, details in video_data.items():
    # Get the word being signed
    word = get_signed_word(details["action"])
    
    # Create the destination folder if it doesn't exist
    destination_folder = os.path.join(destination_root_folder, word)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Construct the video file path
    video_file = os.path.join(videos_folder, f'{video_id}.mp4')  # Assuming videos are all .mp4 files
    missing_video_file = os.path.join(missing_videos_folder, f'{video_id}.mp4')

    # Check if the video exists and copy it to the appropriate folder
    if os.path.exists(video_file):
        destination_file = os.path.join(destination_folder, f'{video_id}.mp4')
        shutil.copy(video_file, destination_file)
        print(f'Copied {video_id}.mp4 to {destination_folder}')
    elif os.path.exists(missing_video_file): # Check if video not found in "videos" checking missing_videos and copy it to the appropriate folder
        destination_file = os.path.join(destination_folder, f'{video_id}.mp4')
        shutil.copy(missing_video_file, destination_file)
        print(f'Copied {video_id}.mp4 to {destination_folder}')
    else: # Video not found in either folder
        print(f'Video {video_id}.mp4 not found in folders')
    video_counter += 1
    
print(f'{video_counter} Videos Processed')
