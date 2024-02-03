import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dlib

def save_and_close_plot(save_path, file_name, format='png'):
    """
    Save the current matplotlib plot and close the figure.

    Parameters:
    - save_path (str): The directory where the image will be saved.
    - file_name (str): The name of the saved image file (without extension).
    - format (str): The format of the saved image (default is 'png').
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the current plot
    file_path = os.path.join(save_path, f"{file_name}.{format}")
    plt.savefig(file_path, format=format)

    # Close the current figure
    plt.close()


def extract_pixel_intensities(image_path, roi_coords):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi = image[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]
    return roi.flatten()

# Function to generate Histograms
def generate_histogram(image_paths, roi_coords, emotions):
    for emotion_code, emotion in zip(['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'], emotions):
        emotion_intensities = []
        for image_path in image_paths:
            if emotion_code in image_path:
                intensities = extract_pixel_intensities(image_path, roi_coords)
                emotion_intensities.append(intensities)

        if emotion_intensities:
            # Plot Histograms
            plt.figure(figsize=(10, 6))
            plt.hist(np.concatenate(emotion_intensities), bins=256, range=(0, 256), density=True, alpha=0.7, color='blue', label='Intensity Distribution')
            plt.title(f'Histogram of Pixel Intensities for {emotion}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Normalized Frequency')
            plt.legend()
            plt.show()
        else:
            print(f"No intensities found for {emotion}")

def generate_mean_intensity_heatmap(image_paths, roi_coords, emotions):
    num_emotions = len(emotions)
    fig, axes = plt.subplots(num_emotions, 1, figsize=(10, 8), sharex=True)

    for i, (emotion_code, emotion) in enumerate(zip(['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'], emotions)):
        emotion_intensities = []
        for image_path in image_paths:
            if emotion_code in image_path:
                intensities = extract_pixel_intensities(image_path, roi_coords)
                emotion_intensities.append(intensities)

        if emotion_intensities:
            mean_intensity = np.mean(emotion_intensities, axis=0)
            axes[i].plot(mean_intensity, label=emotion)
            axes[i].legend()
        else:
            axes[i].set_title(emotion + " (No data)")

    plt.xlabel('Pixel Position')
    plt.suptitle('Mean Intensity Heatmaps')
    plt.show()

def generate_boxplots(image_paths, roi_coords, emotions):
    intensity_data = []

    for emotion_code, emotion in zip(['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'], emotions):
        emotion_intensities = []
        for image_path in image_paths:
            if emotion_code in image_path:
                intensities = extract_pixel_intensities(image_path, roi_coords)
                emotion_intensities.append(intensities)

        if emotion_intensities:  # Check if there are intensities for the emotion
            intensity_data.append(np.concatenate(emotion_intensities))
        else:
            intensity_data.append([])  # Add an empty list if no intensities found

    # Plot Boxplots
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=intensity_data, notch=True)
    plt.title('Boxplots of Pixel Intensities')
    plt.xlabel('Emotion')
    plt.ylabel('Pixel Intensity')
    plt.show()
    
    
# Function to detect facial features using Haar Cascade Classifiers
def detect_features(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyebrow_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_eyepair_small.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

    gray = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.show()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Initialize coordinates for all ROIs
    all_roi_coords = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            roi_coords = [y + ey, y + ey + eh, x + ex, x + ex + ew]
            all_roi_coords.append(roi_coords)

        # Detect eyebrows
        eyebrows = eyebrow_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyebrows:
            roi_coords = [y + ey, y + ey + eh, x + ex, x + ex + ew]
            all_roi_coords.append(roi_coords)

        # Detect mouth
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in mouths:
            roi_coords = [y + ey, y + ey + eh, x + ex, x + ex + ew]
            all_roi_coords.append(roi_coords)

    return all_roi_coords

# Example Usage
image_folder = 'D:\\Side_Projects\\emotion_classification\\datasets\\raw'
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Update with your emotions

# Get list of image paths
# image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.tiff')]

for emotion in emotions:

    image_folder = os.path.join(image_folder, emotion)

    # Perform intensity analysis for each facial image and each ROI
    for image_path in os.listdir(image_folder):
        
        image_path = os.path.join(image_folder, image_path)
        
        print('Analyzing Image: %s' % image_path)
        
        # Detect facial features
        all_roi_coords = detect_features(image_path)

        if all_roi_coords:
            # Display facial image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.title('Facial Image')
            plt.show()

            # Display each ROI
            for roi_coords in all_roi_coords:
                roi = image[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]
                # plt.figure(figsize=(5, 5))
                plt.imshow(roi, cmap='gray')
                plt.title('ROI')
                plt.show()

                # Perform histogram intensity analysis
                # generate_histogram([image_path], roi_coords, emotions)
                
                # # Display Mean Intensity Heatmap
                # generate_mean_intensity_heatmap(image_paths, roi_coords, emotions)

                # # Display Boxplots
                # generate_boxplots(image_paths, roi_coords, emotions)

        else:
            print(f"No facial features detected in {image_path}")
