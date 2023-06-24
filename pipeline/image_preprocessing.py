import os
from IPython.display import Image
import rasterio
import numpy as np

def convert_tif_to_jpg(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            img = Image.open(os.path.join(source_dir, filename))
            img.save(os.path.join(target_dir, filename.replace(".tif", ".jpg")), "JPEG")


def convert_ms_images_to_rgb(input_dir, output_dir):
    # Get a list of all files in the input directory
    files = os.listdir(input_dir)

    # Loop through each file in the input directory
    for i, file in enumerate(files):
        # Check if the file is a .tif image
        if file.endswith(".tif"):
            # Construct the input and output file paths
            input_file_path = os.path.join(input_dir, file)
            output_file_path = os.path.join(output_dir, file)

            # Open the 8-band MS image
            with rasterio.open(input_file_path) as src:
                # Read the individual bands
                red = src.read(4)
                green = src.read(2)
                blue = src.read(1)

                # Normalize the bands to the range [0, 255]
                red = ((red - red.min()) / (red.max() - red.min()) * 255).astype('uint8')
                green = ((green - green.min()) / (green.max() - green.min()) * 255).astype('uint8')
                blue = ((blue - blue.min()) / (blue.max() - blue.min()) * 255).astype('uint8')

                # Stack the bands together to create an RGB image
                rgb = np.stack([red, green, blue], axis=0)

                # Save the RGB image using Rasterio
                profile = src.profile
                profile.update(driver='GTiff', count=3, dtype='uint8')

                with rasterio.open(output_file_path, 'w', **profile) as dst:
                    dst.write(rgb)

            print(f"Converted image saved: {output_file_path}")

    return True
