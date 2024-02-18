import cv2
import dlib
import os
import numpy as np

# Load the face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def process_image(image_path, base_output_folder, emotion, target_size=(200, 200)):
    # Read and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Resize the image to the target size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 3)  # Adjusted upsampling
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract base name without the extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Construct directories for each ROI and the whole image within the specific emotion and image name
        image_output_folder = os.path.join(base_output_folder, emotion, base_name)
        whole_image_dir = os.path.join(image_output_folder, "whole_image")  # Directory for whole images
        eyebrows_dir = os.path.join(image_output_folder, "roi_eyebrows")
        eyes_dir = os.path.join(image_output_folder, "roi_eyes")
        mouth_dir = os.path.join(image_output_folder, "roi_mouth")
        
        # Create directories if they don't exist
        for directory in [whole_image_dir, eyebrows_dir, eyes_dir, mouth_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save the whole image
        cv2.imwrite(os.path.join(whole_image_dir, f"{base_name}_whole_image.jpg"), image)  # Save the original resized image
        
        # Function to crop ROI, display, and save
        def crop_display_save(roi, dir_path, roi_name):
            # Check if the ROI is valid
            if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                cv2.imwrite(os.path.join(dir_path, f"{base_name}_{roi_name}.jpg"), roi)  # Save the ROI image
            else:
                print(f"Invalid ROI for {roi_name}, skipping...")

        # Cropping, displaying, and saving ROIs
        print(f"Processing ROIs for {base_name}")
        crop_display_save(image[min([landmarks.part(n).y for n in range(17, 27)]) - 3:max([landmarks.part(n).y for n in range(19, 22)]) + 3, min([landmarks.part(n).x for n in range(17, 27)]) - 3:max([landmarks.part(n).x for n in range(17, 27)]) + 3], eyebrows_dir, "eyebrows")
        crop_display_save(image[min([landmarks.part(n).y for n in range(36, 48)]) - 3:max([landmarks.part(n).y for n in range(36, 48)]) + 3, min([landmarks.part(n).x for n in range(36, 48)]) - 3:max([landmarks.part(n).x for n in range(36, 48)]) + 3], eyes_dir, "eyes")
        crop_display_save(image[min([landmarks.part(n).y for n in range(48, 68)]) - 3:max([landmarks.part(n).y for n in range(48, 68)]) + 3, min([landmarks.part(n).x for n in range(48, 68)]) - 3:max([landmarks.part(n).x for n in range(48, 68)]) + 3], mouth_dir, "mouth")
        print(f"ROIs Processed and Saved for {base_name}")

# Example usage
input_folder = "D:\\Side_Projects\\emotion_classification\\datasets\\01_occluded"
output_folder = "D:\\Side_Projects\\emotion_classification\\datasets\\obj3_task2_results"
os.makedirs(output_folder, exist_ok=True)

for emotion_folder in os.listdir(input_folder):
    print("Processing emotion:", emotion_folder)
    emotion_path = os.path.join(input_folder, emotion_folder)
    for image_file in os.listdir(emotion_path):
        print("Processing image:", image_file)
        image_path = os.path.join(emotion_path, image_file)
        process_image(image_path, output_folder, emotion_folder)
