import os
import shutil

def create_directory_structure(src_folder, dest_folder, subfolders):
    # Iterate through the source folder
    for emotion_folder in os.listdir(src_folder):
        emotion_path = os.path.join(src_folder, emotion_folder)

        # Skip non-directory items
        if not os.path.isdir(emotion_path):
            continue

        # Iterate through .tiff files in each emotion folder
        for file_name in os.listdir(emotion_path):
            if file_name.endswith(".tiff"):
                # Create corresponding folder in the destination
                file_name_no_ext = os.path.splitext(file_name)[0]
                dest_file_folder = os.path.join(dest_folder, emotion_folder, file_name_no_ext)
                os.makedirs(dest_file_folder, exist_ok=True)

                # Create subfolders within the destination folder
                for subfolder in subfolders:
                    subfolder_path = os.path.join(dest_file_folder, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)

                # Copy .tiff file to destination
                # src_file_path = os.path.join(emotion_path, file_name)
                # dest_file_path = os.path.join(dest_file_folder, file_name)
                # shutil.copy(src_file_path, dest_file_path)

if __name__ == "__main__":
    # Specify source and destination folders
    src_folder = r"D:\Side_Projects\emotion_classification\datasets\JAFFE"
    dest_folder = r"D:\Side_Projects\emotion_classification\datasets\task3_results"

    # Specify subfolders to create within each file folder
    subfolders = ["box_plot", "histogram_intensity", "mean_intensity_heatmaps", "roi_eyebrow", "roi_eyes", "roi_mouth", "whole_images"]

    # Create directory structure and copy files
    create_directory_structure(src_folder, dest_folder, subfolders)

    print("Directory structure created successfully.")
