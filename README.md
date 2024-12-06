# Facemask-detection-OpenCV

A real-time face mask detection web application deployed using Flask. This project utilizes the VGG16 pre-trained model as the base for detecting faces with and without face masks. The model was fine-tuned with 7,000 images (both with and without masks), and various data augmentation techniques were applied to improve its robustness and generalization.

Key Features:

Real-time Face Mask Detection: Detects whether a person is wearing a mask or not from a webcam or uploaded image.
Flask Web Application: Deployed as a web app, allowing users to interact with the model through a simple interface.
Data Augmentation: Applied techniques such as flipping, rotation, and scaling to expand the dataset and enhance model performance.
VGG16 Model: Used as the base model, leveraging its power in image recognition tasks, then fine-tuned for the specific task of mask detection.
High Accuracy: Achieved an accuracy of 96% after fine-tuning the model.
Dataset:

The dataset consists of 7,000 images labeled into two categories:

With Mask
Without Mask
These images were preprocessed and augmented to improve model accuracy and generalization.

Model Training:

VGG16 Base Model: The VGG16 model, pre-trained on the ImageNet dataset, was used as a feature extractor and fine-tuned for the mask detection task.
Data Preprocessing: Images were resized and normalized to match the input requirements of the VGG16 model.
Data Augmentation: Techniques like rotation, horizontal flips, and scaling were applied to enhance the dataset and avoid overfitting.
Fine-tuning: The last few layers of the VGG16 model were fine-tuned with the mask dataset to improve performance, achieving a 96% accuracy.
Technologies Used:

Flask: For deploying the application as a web app.
Keras / TensorFlow: For model building, training, and fine-tuning.
OpenCV: For face detection in real-time video.
Python: The programming language used for building and training the model.
