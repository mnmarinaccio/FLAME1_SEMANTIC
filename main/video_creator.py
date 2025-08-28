import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms 
import torchvision
import albumentations as A
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


# utility functions
def showImage(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return None


def load_and_sort_images(input_folder):
    files = os.listdir(input_folder)
    
    sorted_files = []
    
    for file_name in files:
        try:
            # Attempt to split the filename and convert the second part into an integer
            part = file_name.split('_')[1].split('.')[0]
            num = int(part)  # Attempt conversion
            # If splitting and conversion are successful, add the filename and number to the list of sorted files
            sorted_files.append((file_name, num))
        except (IndexError, ValueError):
            # If splitting or conversion fails, skip the file
            pass
    
    # Sort the list of sorted files based on the extracted numbers
    sorted_files.sort(key=lambda x: x[1])
    
    # Extract only the filenames from the sorted list
    sorted_filenames = [file_name for file_name, _ in sorted_files]
    
    return sorted_filenames

# Usage
input_folder = './dataset/Images'
output_folder = './output'
sorted_filenames = load_and_sort_images(input_folder)
print(sorted_filenames)




if __name__ == "__main__":
    # load in model
    model_name = 'unetc_model'
    model_path = f'./model/{model_name}.pth'
    
    model = torch.load(model_path);
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device);
    model.eval();


    imgCounter=-1
    # Sorted filenames loop through 
    for filename in sorted_filenames:
        imgCounter+=1
        #print(f'Filename = {filename}')
        
        # Open the image using PIL
        img = Image.open(os.path.join(input_folder, filename))
        
        # Show the image
        #showImage(img)
        
        # Apply the transformation
        transform = A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
        transformed_img = transform(image=np.array(img))['image']
        
        # Convert the transformed image back to PIL format
        transformed_img_pil = Image.fromarray(transformed_img)
        
        # Display the transformed image
        #showImage(transformed_img_pil)
        
        # Convert the transformed image to tensor and move to device
        input_tensor = transforms.ToTensor()(transformed_img_pil).unsqueeze(0).to(device)
        #print(f'Input Tensor Shape: {input_tensor.shape}')
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            #print(f'Output Shape: {output.shape}')
            
            output_clamped = torch.clamp(output, 0, 1)
            
            zero_tensor = torch.tensor(0., device=device, requires_grad=True)
            one_tensor = torch.tensor(1., device=device, requires_grad=True)
            output_bin = torch.where(output_clamped < 0.5, zero_tensor, one_tensor)
            
            # get the indices where output_bin == 1
            indices = torch.nonzero(output_bin == 1)
            
            # Extract row and column indices for one image
            row_indices = indices[:, 2]
            col_indices = indices[:, 3]
            
            for row, col in zip(row_indices, col_indices):
                # Change the pixel value to dark blue (0, 0, 139)
                transformed_img_pil.putpixel((col, row), (0, 0, 139))
            
            # Save the modified image
            transformed_img_pil.save(f'{output_folder}/image_{imgCounter}.jpg')
    
        if imgCounter % 20 == 0: 
            print(f'{imgCounter} images processed...')
        # Break loop after processing one image
        #break  
    
    print(f'DONE')
            

