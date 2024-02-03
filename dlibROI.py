import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_intensity_histogram(image, image_folder_path, image_sub_folder_path, image_name):
    intensity_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Intensity Histogram")
    plt.plot(intensity_hist)
    hist_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_histogram.png")
    os.makedirs(os.path.dirname(hist_output_path), exist_ok=True)
    plt.savefig(hist_output_path)
    plt.close()
    print("Intensity Histogram Saved")

def save_intensity_boxplot(image, image_folder_path, image_sub_folder_path, image_name):
    intensity_values = image.flatten()
    plt.figure()
    plt.title("Intensity Box Plot")
    sns.boxplot(intensity_values)
    boxplot_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_boxplot.png")
    os.makedirs(os.path.dirname(boxplot_output_path), exist_ok=True)
    plt.savefig(boxplot_output_path)
    plt.close()
    print("Box Plot Saved")

def save_mean_intensity_heatmap(image, image_folder_path, image_sub_folder_path, image_name):
    mean_intensity = np.mean(image, axis=2)
    plt.figure()
    plt.title("Mean Intensity Heatmap")
    sns.heatmap(mean_intensity, cmap="viridis", cbar=True)
    heatmap_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_heatmap.png")
    os.makedirs(os.path.dirname(heatmap_output_path), exist_ok=True)
    plt.savefig(heatmap_output_path)
    plt.close()
    print("Mean Intensity Heatmap Saved")



def detect_features(image_path):
    print("Detecting Features from Image:", image_path)  # Debug
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (580,580))
    
    plt.imshow(gray, cmap='gray')
    plt.title('Displayed Image')
    plt.show()

    # Initialize dlib face detector
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    # Initialize dlib shape predictor for 68 face landmarks
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Initialize coordinates for eyebrows, eyes, and mouth
    left_eyebrow_coords = []
    right_eyebrow_coords = []
    left_eye_coords = []
    right_eye_coords = []
    mouth_coords = []

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract coordinates of eyebrows
        left_eyebrow_coords.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)])
        right_eyebrow_coords.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)])

        # Extract coordinates of eyes
        left_eye_coords.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_coords.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Extract coordinates of mouth
        mouth_coords.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

    return (
        left_eyebrow_coords, right_eyebrow_coords,
        left_eye_coords, right_eye_coords,
        mouth_coords,
        image
    )

def process_image(image_path, output_folder):
    print("Processing Image:", image_path)  # Debug

    (
        left_eyebrow_coords, right_eyebrow_coords,
        left_eye_coords, right_eye_coords,
        mouth_coords,
        image
    ) = detect_features(image_path)

    print("Features Detection Completed")

    # Get the image name (excluding the file extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create a folder for the current image within the emotion-specific folder
    image_folder_path = os.path.join(output_folder, image_name)
    os.makedirs(image_folder_path, exist_ok=True)

    # print("Saving Original Image")
    # image_sub_folder_path = "whole_image"
    # # Save the original image in the "whole_image" folder
    # whole_image_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + ".png")
    # os.makedirs(os.path.dirname(whole_image_output_path), exist_ok=True)  # Create the directory if it doesn't exist
    # cv2.imwrite(whole_image_output_path, image)
    # print("Original Image Saved Successfully")
    
    # print("Generating and Saving Histogram")
    # save_intensity_histogram(image, image_folder_path, image_sub_folder_path, image_name)
    # print("Generating and Saving Box Plot")
    # save_intensity_boxplot(image, image_folder_path, image_sub_folder_path, image_name)
    # print("Generating and Saving Heatmap")
    # save_mean_intensity_heatmap(image, image_folder_path, image_sub_folder_path, image_name)

    #!------------------------------------------------------#

    print("Processing Eyebrows Coordinates")
    # Combine both left and right eyebrow coordinates
    all_eyebrow_coords = left_eyebrow_coords + right_eyebrow_coords

    if all_eyebrow_coords:

        # Get bounding box for all eyebrows with spacing
        margin = 1
        x_vals, y_vals = zip(*all_eyebrow_coords)
        min_x, max_x = min(x_vals) - margin, max(x_vals) + margin
        min_y, max_y = min(y_vals) - margin, max(y_vals) + margin

        print("Cropping Eyebrows")
        # Crop the region containing both eyebrows with spacing
        eyebrows_roi = image[max(0, min_y):min(max_y, image.shape[0]), max(0, min_x):min(max_x, image.shape[1])]
        print("Eyebrows Cropped Successfully")

        # Save the cropped eyebrows as .png using matplotlib
        image_sub_folder_path = "roi_eyebrow"
        eyebrows_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_eyebrows.png")
        os.makedirs(os.path.dirname(eyebrows_output_path), exist_ok=True)  # Create the directory if it doesn't exist
        print("Saving Cropped Eyebrows Image to ", eyebrows_output_path)
        # plt.imshow(cv2.cvtColor(eyebrows_roi, cv2.COLOR_BGR2RGB))
        # plt.savefig(eyebrows_output_path)
        # plt.close()
        cv2.imwrite(eyebrows_output_path, eyebrows_roi)
        print("Cropped Eyebrows Image Saved")
        
    
        # print("Generating and Saving Histogram")
        # save_intensity_histogram(eyebrows_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Box Plot")
        # save_intensity_boxplot(eyebrows_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Heatmap")
        # save_mean_intensity_heatmap(eyebrows_roi, image_folder_path, image_sub_folder_path, image_name)
    
    else:
        print("Eyebrows not detected")
    
    #!--------------------------------------------------------#

    print("Processing Eyes Coordinates")
    # Combine both left and right eye coordinates
    all_eye_coords = left_eye_coords + right_eye_coords

    if all_eye_coords:

        # Get bounding box for all eyes with spacing
        margin = 3
        x_vals, y_vals = zip(*all_eye_coords)
        min_x, max_x = min(x_vals) - margin, max(x_vals) + margin
        min_y, max_y = min(y_vals) - margin, max(y_vals) + margin

        print("Cropping Eyes From Image")
        # Crop the region containing both eyes with spacing
        eyes_roi = image[max(0, min_y):min(max_y, image.shape[0]), max(0, min_x):min(max_x, image.shape[1])]

        # Save the cropped eyes as .png using matplotlib
        image_sub_folder_path = "roi_eyes"
        eyes_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_eyes.png")
        os.makedirs(os.path.dirname(eyes_output_path), exist_ok=True)  # Create the directory if it doesn't exist
        print("Saving Cropped Eyes Image to ", eyes_output_path)
        # plt.imshow(cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2RGB))
        # plt.savefig(eyes_output_path)
        # plt.close()
        cv2.imwrite(eyes_output_path, eyes_roi)
        
            
        # print("Generating and Saving Histogram")
        # save_intensity_histogram(eyes_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Box Plot")
        # save_intensity_boxplot(eyes_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Heatmap")
        # save_mean_intensity_heatmap(eyes_roi, image_folder_path, image_sub_folder_path, image_name)
        
    else:
        print("Eyes not detected") 
        
    #!--------------------------------------------------------#

    print("Processing Mouth Coordinates")
    # Combine all mouth coordinates
    all_mouth_coords = mouth_coords

    if all_mouth_coords:
        
        # Get bounding box for the mouth with spacing
        margin = 3
        x_vals, y_vals = zip(*all_mouth_coords)
        min_x, max_x = min(x_vals) - margin, max(x_vals) + margin
        min_y, max_y = min(y_vals) - margin, max(y_vals) + margin

        print("Cropping Mouth Image")
        # Crop the region containing the mouth with spacing
        mouth_roi = image[max(0, min_y):min(max_y, image.shape[0]), max(0, min_x):min(max_x, image.shape[1])]

        # Save the cropped mouth as .png using matplotlib
        image_sub_folder_path = "roi_mouth"
        mouth_output_path = os.path.join(image_folder_path, image_sub_folder_path, image_name + "_mouth.png")
        os.makedirs(os.path.dirname(mouth_output_path), exist_ok=True)  # Create the directory if it doesn't exist
        print("Saving Mouth Image to ", mouth_output_path)
        # plt.imshow(cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB))
        # plt.savefig(mouth_output_path)
        # plt.close()
        cv2.imwrite(mouth_output_path, mouth_roi)
            
        # print("Generating and Saving Histogram")
        # save_intensity_histogram(mouth_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Box Plot")
        # save_intensity_boxplot(mouth_roi, image_folder_path, image_sub_folder_path, image_name)
        # print("Generating and Saving Heatmap")
        # save_mean_intensity_heatmap(mouth_roi, image_folder_path, image_sub_folder_path, image_name)
        
    else:
        print("Mouth not detected")    
        
    #!--------------------------------------------------------#


# Specify input and output folders
input_folder = "D:\\Side_Projects\\emotion_classification\\datasets\\raw"
output_folder = "D:\\Side_Projects\\emotion_classification\\datasets\\task6_results"

# List of emotion folders
emotion_folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Loop through each emotion folder
for emotion_folder in emotion_folders:
    emotion_folder_path = os.path.join(input_folder, emotion_folder)

    # Create output subfolders for each emotion
    output_subfolder = os.path.join(output_folder, emotion_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Loop through each .tiff file in the emotion folder
    for file_name in os.listdir(emotion_folder_path):
        # if file_name.endswith(".jpg"):
        image_path = os.path.join(emotion_folder_path, file_name)

        # Process the image and save features
        process_image(image_path, output_subfolder)

print("Processing completed.")
