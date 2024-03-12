# Synthetic Bounding Box Dataset Generator

## Introduction
This project provides a provisional solution for generating synthetic image datasets with bounding box annotations, offering an alternative to the cumbersome and resource-intensive process of creating large-scale image datasets manually. By superimposing foreground objects onto background images and automatically generating corresponding bounding box annotations, this tool facilitates rapid dataset generation for various computer vision tasks, particularly useful in the context of object detection model training.

## How It Works
The generator leverages a dataset consisting of separate foreground and background images. It randomly selects a foreground object and a background scene, superimposes the object onto the background at a random position, and calculates the bounding box coordinates for the placed object. The output is a modified image with the object integrated into the scene and its corresponding bounding box annotation.

Key Features:
- Random selection of backgrounds and foregrounds to create diverse scenarios.
- Dynamic positioning of foreground objects within the background boundaries.
- Automatic generation of bounding box annotations corresponding to the object's location.
- Configurable probability for edge blending to improve the realism of superimposed objects.

## Quick Start Example

1. **Prepare your environment:**
   Ensure you have Python 3.11 installed and create a virtual environment for the project:

```sh
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

2. **Install dependencies:**
Clone the repository and install the required Python packages:

```sh
git clone https://your-repository-url.git
cd your-project-directory
pip install -r requirements.txt
```

3. **Organize your dataset:**
Arrange your foreground and background images in the respective `foregrounds` and `backgrounds` folders within a main dataset directory:

dataset/  
├── backgrounds/  
│ ├── background1.jpg  
│ ├── background2.jpg  
│ └── ...  
└── foregrounds/  
├── object1.png  
├── object2.png  
└── ...

4. **Use the dataset generator:**
Instantiate and utilize the dataset generator in your Python script:

```python
from synthetic_dataset import SuperimposeDataset
from torchvision.transforms import ToTensor

# Initialize dataset
dataset_path = 'path/to/your/dataset'
transform = ToTensor()
dataset = SuperimposeDataset(dataset_path, transform=transform)

# Fetch a sample
image, bbox = dataset[0]  # Get the first generated sample
print(f"Image shape: {image.size()}, Bounding box: {bbox}")
```

By following these steps, you can quickly generate a synthetic dataset tailored to your specific needs, facilitating the development and training of object detection models without the need for extensive manual annotation work.

This README file provides an overview of the project, instructions for setting it up, and a simple example of how to generate a dataset. Modify the repository URL and any specific instructions according to your project setup and requirements.
