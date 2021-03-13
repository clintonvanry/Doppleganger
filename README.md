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

![](https://github.com/clintonvanry/Doppleganger/blob/main/resnet.png)

As most CNN architectures, ResNet contains a bank of Convolutional (Conv) Layers followed by a Fully Connected (FC) Layer.

The bank of conv layers produce a feature vector in higher dimensional space just like the HOG descriptor. So what’s the difference between a bank of conv layers and HOG descriptor? The most important differences are

HOG is a fixed descriptor. There is an exact recipe for calculating the descriptor. On the other hand, a bank of conv layers contains many convolution filters. These filters are learned from the data. So unlike HOG, they adapt based on the problem at hand.

The FC layer does the same job as the SVM classifier in traditional approaches. It classifies the feature vector. In fact, sometimes the final FC layer is replaced by an SVM.

Any image can be vectorized by simply storing all the pixel values in a tall vector. This vector represents a point in higher dimensional space. However, this space is not very good for measuring distances. In a face recognition application, the points representing two different images of the same person may be very far away and the points representing images of two different people may actually be close by.

When we used PCA to reduce dimensionality, we hoped distances in this reduced dimensional space would be more meaningful. Similarly, with Fisher’s Linear Discriminant we tried to find a space where distances were meaningful. Both approaches work to a certain extent but the performance is by no means exceptional.

Deep Metric Learning is a class of techniques that uses Deep Learning to learn a lower dimensional effective metric space where images are represented by points such that images of the same class are clustered together and images of different class are far apart. Conceptually, the goals are very similar to Fisher’s Linear Discriminant but in practice the results are vastly superior because instead of directly reducing the dimension of the pixel space, the convolution layers first calculate the meaningful features which are then implicitly used to create the metric space.

Turns out we can use the same CNN architecture we use for image classification for deep metric learning.
You input an image and the output is a point in 128 dimensional space. If you want to find how closely related two images are, you can simply find the pass both images through the CNN and obtain the two points in this 128 dimensional space. You can compare the two points using simple L2 ( Euclidean ) distance between them.

In order to use the Resnet CNN we need to train it with images from the celeb dataset. 

This process is called ennrollment and has the following steps:
1. Define the network
  - This defines the ResNet neural network used for training the model. The first few layers are convolutional layers and the final layer is the loss metric
2. Load the model for face landmakrs and face recognition
  - Initialize Dlib’s Face Detector, Facial Landmark Detector and Face Recognition neural network objects
3. Process each image in the dataset and compute descriptors
  - Process enrollment images one by and one.
  - Convert image from RGB to BGR, because Dlib uses BGR as default format.
  - Detect faces in the image. For each face we will compute a face descriptor.
  - For each face get facial landmarks.
  - Compute face descriptor using facial landmarks. This is a 128 dimensional vector which represents a face.
  - For each face descriptor we will also save the corresponding label
4. Save the mapping between face ID and person
5. Save the updated model  
  - Now save descriptors and the descriptor-label mapping to disk.
 
code example:

`void enrollment(std::map<int, std::string>& labelNameMap, std::vector<matrix<float,0,1>>& faceDescriptors, std::vector<int>& faceLabels,
                frontal_face_detector& faceDetector, const shape_predictor& landmarkDetector, anet_type& net, std::map<int,std::string>& folderImageMap)
{

    std::vector<std::string> names;
    std::vector<int> labels;

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<std::string> imagePaths;
    std::vector<int> imageLabels;
    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folderNames;
    getFolderAndFiles(names,labels,imagePaths,labelNameMap,imageLabels);

    // process training data
    // We will store face descriptors in vector faceDescriptors
    // and their corresponding labels in vector faceLabels
    //std::vector<matrix<float,0,1>> faceDescriptors;
    //std::vector<int> faceLabels;

    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++)
    {
        std::string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];

        std::cout << "processing: " << imagePath << std::endl;

        if(!mapContainsKey(folderImageMap,imageLabel))
        {
            folderImageMap[imageLabel] = imagePath;
        }

        // read image using OpenCV
        Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

        // convert image from BGR to RGB
        // because Dlib used RGB format
        Mat imRGB;
        cvtColor(im, imRGB, COLOR_BGR2RGB);

        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

        // detect faces in image
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        // Now process each face we found
        for (int j = 0; j < faceRects.size(); j++) {

            // Find facial landmarks for each detected face
            full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);

            // object to hold preProcessed face rectangle cropped from image
            matrix<rgb_pixel> face_chip;

            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            matrix<float,0,1> faceDescriptor = net(face_chip);

            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels
            faceDescriptors.push_back(faceDescriptor);
            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            faceLabels.push_back(imageLabel);
        }
    }

    std::cout << "number of face descriptors " << faceDescriptors.size() << std::endl;
    std::cout << "number of face labels " << faceLabels.size() << std::endl;
}`












