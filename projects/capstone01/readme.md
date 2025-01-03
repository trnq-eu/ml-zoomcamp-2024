# Architectural Style Classification Project

This project aims to develop an image classification model to identify the architectural style of buildings. The model leverages **transfer learning** based on the **Xception** pre-trained model to achieve high accuracy in recognizing various architectural styles.

## Problem Description
The goal of this project is to create a machine learning model capable of recognizing the architectural style of a building from an image. Architecture is a rich and diverse field, with styles ranging from Gothic to Modernist, each with distinct features and characteristics.

Identifying these styles can be a complex task, requiring expertise and knowledge of architectural elements. This project aims to simplify this process by leveraging computer vision and machine learning techniques to automate the recognition of architectural styles.

Such a model can have applications in various fields, including cultural heritage preservation, urban planning, education, and even enhancing user experiences in applications like virtual tours or travel guides.

## Files
* **notebook.ipynb**: contains the Jupyter notebook with the dataset creation process, the exploratory data analysis and the training of different models with optimized parameters
* **train.py**: contains the code for the training of the final model
* **app.py**: contains a Flask API to make predictions using the final model
* **Pipfile** and **Pipfile.lock**: files for dependencies management
* The **models** folder contains the final trained model
* The **pictures** folder contains some random pictures to test the model
* **Dockerfile** contains the instructions to build and run the Docker container

## Deployment instructions
Build and run a Docker container to deploy and run the model:

1. **Install Docker:**
   Make sure Docker is installed on your system. Follow instructions for your operating system from the official Docker website: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

2. **Build the Docker Image:**
```bash
docker build -t classify-architecture .
```


3. **Run the Docker Container**:
```bash
docker run -p 5000:5000 classify-architecture
```

4. **Test the model**:

Once the Docker container is up and running you can test the model by submitting an image with curl:

```bash
curl -X POST -F "image=@./pictures/random_building_1.jpeg" http://localhost:5000/predict
```


## Dataset Creation

Instead of using an existing dataset, I created our own dataset by downloading images from Wikimedia Commons. I used the following categories:

- Arabic architecture
- Baroque architecture
- Brutalist architecture
- Colonial architecture
- Gothic architecture
- Modernist architecture
- Postmodern architecture
- Byzantine architecture
- Art Deco architecture
- Beaux Arts architecture
- Palladian architecture

Inside the `notebook.ipynb` file you can find the script used to download the images using the Wikimedia API to fetch all images belonging to a specific category page. 
Each category is saved into a separate folder.

I also integrated an existing dataset from Kaggle ([https://www.kaggle.com/datasets/wwymak/architecture-dataset/data](https://www.kaggle.com/datasets/wwymak/architecture-dataset/data)) to enhance and expand the dataset.

## Data Cleaning

- Some images were converted to JPEG format using OpenCV to ensure compatibility and handle different file formats.
- Manual inspection was performed to remove images that were not suitable for training, such as those not depicting buildings.

## Exploratory Data Analysis (EDA)

- **Image Gallery:** A utility function was implemented to display a gallery of images with class labels.
- **File Format Check:** A function was used to check the format of all images.
- **Directory Structure Analysis:** Analysis of the folders revealed:
    - 11 classes of architectural styles.
    - The total number of images in the dataset.
    - The distribution of images across different classes.
- **Class Imbalance:** The dataset has class imbalance issues, with some styles having significantly more images than others.
- **Image Dimensions:** The average, maximum, and minimum widths and heights of images were calculated.
- **Oversized Images:** Identified and resized images that were significantly larger than the average image size.

## Training

- **Data Loading:** Keras `ImageDataGenerator` was used to load the dataset, split it into training and validation sets (80%/20%).
- **Transfer Learning:**
    - Pre-trained Xception model was used as the base.
    - The base model was frozen to act as a feature extractor.
    - Custom layers were added on top of the pre-trained model for classification.
- **Optimization:**
    - The learning rate was tuned by testing different learning rates and plotting validation accuracy. A learning rate of 0.001 was selected.
    - Model checkpointing was implemented to save the best performing model based on validation accuracy during training.
- **Model Enhancement:**
    - Added a fully connected inner layer to the model and tested different size of this layer.
    - Dropout layers for regularization were introduced and the best dropout value was found.
- **Data Augmentation:**
    - Different data augmentation techniques (e.g. rotation, shifts, rescaling, flips) provided by `ImageDataGenerator` were used to increase the dataset size and improve generalization.
    - Augmented images were visualized to verify the augmentation strategy.
- **Training with Augmented Dataset:** The model was trained using the augmented dataset.
- **Larger Model:** A larger input size (299x299) was tested to see if there was an improvement of accuracy.

## Model Prediction

- The best model was loaded and evaluated on the validation set.
- A custom function was written to test and predict the category of an input image.

## Deployment

- Inside the notebook, the trained model was converted to a TensorFlow Saved Model format.
- The saved model was inspected to ensure the correct input and output signatures.
- Due to library incompatibility issues, I had to renounce to deploy the model using Tensorflow Serving. I created a Flask API using the original .keras model instead (see `app.py` file)

Here you can find a screenshot of the deployed model 