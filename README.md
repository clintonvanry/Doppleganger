# Doppleganger project
In this project, we will build a Fun Application - We will use the Face Embeddings to find a doppelganger or look-alike to a given person. Specifically, we will give you images of two persons and a celebrity dataset. Your task will be to find their celebrity look-alike. The first person is Sofia Solares who looks like the American Singer Selena Gomez and the second one is Shashikant Pedwal who looks like Indian Film Actor Amitabh Bachchan.
You simply need a dataset which has enough celebrity faces and use face embeddings to match the test image with the celebrity face embeddings

## The Dataset
There are many datasets that contain images of celebrities. Some of them are:

- CelebA Dataset
- VGGFace
- VGGFace2
- MS-Celeb-1M
- Celebrity-Together Dataset

## The Solution

### libraries used in the solution
- OpenCV
- Dlib
- Dlib models
  - face detections: shape predictor
  - face recognition: resnet

### Technique
We will be using deep learning with Dlib.
In a traditional image classification pipeline, we converted the image into a feature vector ( or equivalently a point) in higher dimensional space. This was done by calculating the feature descriptor (e.g. HOG) for an image patch. Once the image was represented as a point in higher dimensional space, we could use a learning algorithm like SVM to partition the space using hyperplanes that separated points representing different classes.

Even though on the surface Deep Learning looks very different from the above model, there are conceptual similarities. Figure 2 reveals the Deep Learning module used by Dlib’s Face Recognition module. The architecture is based on a popular network called ResNet.


