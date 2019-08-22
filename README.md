# Emotions-Recognition
Emotions recognition by using facial images for real-time aircraft systems


% Goal

  The goal of this project was to implement a model for recognizing human emotions given a real
  time facial images with having an accuracy of at least 65%, trained on the dataset of FER2013
  images from kaggle.


% Description

  Facial expressions play a key role in determining human’s feelings and emotions. In aircraft
  systems, determining the pilot’s expressions can be very useful to take initial actions for safety
  concerns and for automating the aircraft’s processes. For recognizing human emotions, plenty
  of research has been done and has been used for various applications in recommendation
  systems, customer remarks, decision making, and in safety critical systems like in avionics.
  Emotions identification can be used for knowing pilot’s current feelings so that if pilot is in
  undesirable feeling for aircraft flying then immediate actions can be made to prevent any big
  catastrophe and any recommendations can be given for better flying and help in decision
  making. Seven elementary categories of human emotions are unanimously predictable across
  different cultures and by numerous people, these are: anger, disgust, fear, happiness, sadness,
  surprise and neutral. These emotions are found to be more difficult to recognize as facial
  expressions for these emotions does not vary so much and sometimes even human cannot
  recognize them.


% Model Explanation

  Model is trained on the dataset of FER2013 dataset containing about 36000 facial images
  labeled into 7 emotions (0 to 6). 80% of the dataset is assigned for training set and 20% for
  testing. Database contain images of various intensities, angles and people of different ages.
  
  ![image](https://user-images.githubusercontent.com/35194791/63518088-b5fac980-c509-11e9-8bf7-a5bdf4e707f3.png)
  
  I have used Convolutional Neural Networks (CNN) to train the model on these images, its
  architecture is as follows:
  
  ![image](https://user-images.githubusercontent.com/35194791/63518130-ce6ae400-c509-11e9-8fd3-def0e1a8c48f.png)
  
  The input image to the model is of dimensions 48x48 having a single gray scale channel. Input
  image passes to the first layer having kernel size of 5x5 with 64 fliters to learn and uses the
  RELU activation function and applies max pooling of size 3x3. This set of max pool and
  convolutional layer is applied again to the output from previous layer input and finally a layer
  of 4x4 filter size and 128 neuron nodes is applied. Final softmax layer contains 7 nodes with
  activation of softmax function to get the probability output for prediction of each class. Model
  is trained on the 20 epochs with batch size of 128.


% Model Usage

  The model is saved once its trained on the dataset. For recognizing the emotion from the image
  we first preprocess the actual image to convert it into single channel (gray scale) and the face
  is cropped from the image by using viola jones algorithm. The final preprocessed and cropped
  image is passed to the model which gives the prediction for each class (total of 7 classes) and
  finally the class with maximum probability is recognized to be the emotion for the input facial
  image.


% Final Results
  
  The model was tested on 20% images of the FER2013 dataset. The training accuracy of the
  model for 20 epochs is 89.61% having the loss of 25.42% whereas the test accuracy for the
  model is 87.68% with the loss of 30.65%. Model was able to reach the previously defined goal
  (to achieve the accuracy of at least 65%) and could also recognize the emotions correctly for
  many real time facial images.
  

% Dataset Labels

  0: Angry<br/>
  1: Disgust<br/>
  2: Fear<br/>
  3: Happy<br/>
  4: Sad<br/>
  5: Surprise<br/>
  6: Neutral<br/>
  
  ![image](https://user-images.githubusercontent.com/35194791/63518331-3e796a00-c50a-11e9-8e50-c12bfe621176.png)
  ![image](https://user-images.githubusercontent.com/35194791/63518407-6668cd80-c50a-11e9-809f-26deafcf22ac.png)

