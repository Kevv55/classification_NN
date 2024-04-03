# classification_NN
This project uses the Tensorflow framework to create a Neural Network program that classifies the clothing article given an image.ar
The model uses a standard training data and testing data to calculate its training and testing accuracy.
The training accuracy was 91% and testing accuracy 87% which does show some overfitting by the model. 
If you take a look at the image classification performace its predictive performance is very good.

The clothing categories are placed in an array, here is what each index represents:
0: 'T-shirt/top',
1: 'Trouser', 
2: 'Pullover', 
3: 'Dress', 
4: 'Coat',
5: 'Sandal', 
6: 'Shirt', 
7: 'Sneaker', 
8: 'Bag', 
9: 'Ankle boot'

Explanation of the model:
This is a 28x28 pixel image of a boot.
![bootSimple](https://github.com/Kevv55/classification_NN/assets/100497778/8911436f-bcdc-4256-a2f0-d3ad1c68fb16)

The following image shows the approach taken by the model. Here we can see the pixelization of the boot. The model uses this pixelization to learn patterns and make its predictions of which clothing category the boot belongs to.
![Image_pixelization](https://github.com/Kevv55/classification_NN/assets/100497778/58cb8112-480d-4bac-be72-9d79e29af840)

After running the program using a CategoricalCross entrogy loss function, a simple Adam optimizer. In this model we use an input layer which flattens the images, oen fully connected function that uses a RLU activation function and 10 output layers each representing the probability of the output belonging to the clothing category
https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

After the training is done. This is how confident the model is that the given image which it had never seen before is a boot:
![Boot_graph](https://github.com/Kevv55/classification_NN/assets/100497778/b199bfc3-542f-4eef-a55f-c085209d15f7)

Here is the confidence level for other images which were never seen before:
![classification_performance](https://github.com/Kevv55/classification_NN/assets/100497778/b8b2a849-36b4-48b1-985a-e72765cb5c9d)
