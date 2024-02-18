import os
import cv2
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
    sns.boxplot(x=intensity_values)
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

def process_image(image_path, image_folder_path, image_sub_folder_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Assuming the image is in color for the mean intensity heatmap
    if len(image.shape) != 3 or image.shape[2] != 3:
        print(f"Image is not in expected color format: {image_path}")
        return

    save_intensity_histogram(image, image_folder_path, image_sub_folder_path, image_name)
    save_intensity_boxplot(image, image_folder_path, image_sub_folder_path, image_name)
    save_mean_intensity_heatmap(image, image_folder_path, image_sub_folder_path, image_name)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Correctly check for both .jpg and .jpeg files, ignoring case
            if file.lower().endswith(('.jpg', '.jpeg')):  # Note the tuple for multiple extensions
                image_path = os.path.join(root, file)
                image_folder_path = os.path.dirname(root)
                image_sub_folder_path = os.path.basename(root)
                print(f"Processing {image_path}")
                process_image(image_path, image_folder_path, image_sub_folder_path)


# Set the base directory where the images are stored
base_directory = 'D:\\Side_Projects\\emotion_classification\\datasets\\obj3_task2_results'
process_directory(base_directory)

print("Processing complete.")
