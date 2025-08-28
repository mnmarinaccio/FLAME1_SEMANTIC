import os
import imageio

if __name__ == "__main__":
    # Path to the folder containing images
    images_folder = './output'
    
    # Path to save the output video
    output_video_path = 'output_video_UNETmodel.mp4'
    
    files = os.listdir(images_folder)
    
    # List all image files in the folder
    #image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
    image_files = sorted([f for f in files if os.path.isfile(os.path.join(images_folder, f)) and f.endswith('.jpg')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and '.' in x.split('_')[1] else float('inf'))
        
    # Create a list to hold the image data
    images = []
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        # Read each image and append it to the list
        images.append(imageio.imread(image_path))
    
    # Write the images to a video file
    imageio.mimwrite(output_video_path, images, fps=24)
    
    print(f'Video saved at: {output_video_path}')

    