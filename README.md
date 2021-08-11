Histopathology Final Project
====
Introduction
====
In this project, the [Histopathologic Data Set](https://www.kaggle.com/c/histopathologic-cancer-detection) from Kaggle as a basis for exploring the capabilities of Convolutional Neural Networks. The primary goal in this project is to identify the model architecture which can most accurately predict whether an image of lung or colon tissue signifies the presence of cancerous cells.

Background
====

[Efrem will include background here]


Goals
====
Some goals we had in mind in our search for the most optimal modeling pipeline were as follows:

1. Explore: Given a list of predetermined CNN models (Mobilenet, Densenet, Alexnet, Resnet), which will give the best base results?
2. Refine: How can different approaches for the same model yield different approaches? In this step, we will identify whether it's more appropriate to use a single classification Neural Network for both lung and colon cancer images or create two separate models for these regions of the body.
3. Fine Tune: Once we've settled on a baseline model and establised a most viable architecture, how can we tweak hyperparameters such as learning rate and pretrained layer parameters increase the accuracy of predictions?
4. Apply Regularization: Taken into account in parallel with the fine-tuning phase, what are some additional tehniques which can prevent the chosen model from overfitting, and how effective are they at improving the validation accuracy of the model?

Explore: Model Selection
====
The first task in the journey to an optimal model is selecting the most appropriate pretrained model. For this part, we will ask the model to predict one of five classes from the image over five epochs: lung_n, lung_aca, lung_scc, colon_aca, or colon_n. While not entirely related to the question at hand, this will give us a good idea of the way a model behaves on medical images and provide the opportunity for more experience in interpreting Cross Entropy loss. The pretrained layers will be frozen for the moment, and the final 1000 class dense linear layer will instead be replaced with a 5 class linear layer for purposes of multi-class classification. The models tested are as follow:

a. [DenseNet](https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow) 
---- 

Created to remedy the vanishing/exploding gradient problem of deep neural networks, this architecture seeks to retain information that might otherwise be lost to backpropogation when tweaking the parameters in the layers. It relies on concatenating the output of previous layers with future ones, of which presents the downside of being computationally expensive.

![DENSENET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/densenetloss.png)

While densenet may be suitable for many purposes, we see that in this case it's increasingly overfitting as the gap widens between the training and validation loss. Not shown is the accuracy, which relayed an impressive 0.8 for the training set but an abysmal 0.02 for validation. While this overfitting can be dealt with through regularization techniques, it may be best to survey the others before making that leap.

b. [AlexNet](https://en.wikipedia.org/wiki/AlexNet) 
----
While not the first GPU-implementations of a CNN, this model was one of the first to publicize the technique and was proposed in one of the most cited machine learning papers on Google Scholar. It makes use of five convolutional layers connected via three interdispersed maxpool layers, with three fully connected dense layers before outputting 1 of 1000 classes. This is seen as a precursor to the other pretrained models discussed in this section. The idea in choosing this is that, while deep, is in a sweet spot in parameter count and thus has the capacity to give good results while not being too computationally expensive.

![ALEXNET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/alexnetloss.png)

While the initial glance at the losses are very promising, further inspection shows similar lackluster results as DenseNet with respect to overfitting the training set. We do see the validation and training loss converging to a similar scale, which overtakes DenseNet as the best candidate so far.

c. [MobileNet](https://arxiv.org/abs/1704.04861) 
---- 
This model was proposed as a way to implement bulky CNN architecture onto lightweight mobile applications. It does so by introducing hyperparameters to the model which trades off accuracy for computability. For the purposes of this experiment, we have opted to take advantage of the less bulky model to see how it compares to the bigger architectures shown previously:

![MOBILENET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/mobilenetloss.png)

Initial inspection of this model actually shows it performing quite well in terms of closing the loss gap as well as showing a variation in learning as the last layer's parameters are tweaked for the cancerous cells. This gives us hope that looking at the accuracy will give us an ideal model to work with for this project.

![MOBILENET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/mobilenetacc.png)

It looks like we've been duped again! As can be seen above, the validation accuracy is incredibly low, despite the training set reaching pretty high accuracy levels.


d. [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) 
---- 
Drawing influence from the antomical structure of pyramidal cells in the cerebral cortex, ResNets make use of skip connections to feed information from previous layers to future layers. This can be likened to the DenseNet structure, though the layers are fed forward in a more sparse manner. This creates a happy medium between information retention and computation time, which can also be seen as a form a regularization of our system.

![RESNET_LOSS](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/resnetloss.png)
![RESNET_ACC](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/resnetaccs.png)

With a promising close in the gaps appearing in the losses and a gradual improvement of validation accuracy as the number epochs increases, we conclude that ResNet holds the best prospects of providing an accurate classifier once trained and fine-tuned.


Model Architecture
====

Because the goal here is to accurately identify cancer in an image, we want to investigate two strategies with our pretrained model.

Strategy 1:
----
Using the whole entire dataset, we want to investigate whether or not cancerous cells exist in the image regardless of if the image is that of the lung or the colon. This translates to a single binary classification model whose results are shown below:

![Single Model Accuracy](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/cancer_identification_accuracy.png)

This shows very promising initial results, as the accuracy for the training set continually improves while validation accuracy actually surpasses at the sixth epoch. There is good reason to believe that this may be the best way to address our question, but let's take a look at the second strategy before continuing forward.


Strategy 2:
----
Despite the promising results, it may be more worthwhile to train one binary classifier on colon cancer and another binary classifier on lung cancer. When this is done, the following results are achieved:

[INSERT DUAL MODEL ACCURACY HERE!]

[INSERT ANALYSIS OF THE MODEL ACCURACIES HERE]


Fine Tuning
====

With the pretrained model decided and the most effective model architecture chosen, we will finally fine tune hyperparameters and take into consideration some of the regularization constraints to make the model ideal.

1. Data Augmentation - Before tweaking the parameters/hyperparameters themselves, it's a good idea to first tweak the images with augmentation methods before feeding them into the model for training. By inheriting from the Dataset class, we implemented physical transformations such as rotation and reflection, random cropping, saturation, and fuzziness to the images as they were passed along in each epoch. The goal here is to generate new data which the computer sees as novel. This also introduces a meta-geometric aspect that the model can learn in addition to the contents within the image.
2. Adjusted Learning Rate - Instead of relying on a single learning rate, we use Triangular Rates via the cosine function to fluctuate the data. In addition, the learning rate is scaled such that it's extremely small for earlier layers which may not benefit from anything other than the most minute parameter changes.
3. Unfreezing Gradients - As the number of epochs increase, it could be a good idea to unfreeze pretrained layers so that they are free to (minutely) change when the later layers lock into the ideal values.
4. Weight Decay - Applying L2 regularization in the optimizer could be an easy and elegant way to generalize the Neural Network to the validation set.
5. Early Stopping - When the number of epochs begin to detrimentally affect the accuracy of our model, stop training (or in our case, select the saved model with the best validation accuracy throughout epochs).

Conclusion
====
With all of the above techniques applied to the neural network, the following results were found:

[LOSS/ACCURACY GRAPHS]


Looking back on the fine tuning principles, data augmentation seemed to help the most, which aligns with the general notion that a model can only do as well as the data provided. Unfreezing the gradients too quickly tended to result in extreme overfitting and ruined the gains that training made in the validation error, making that a finicky aspect to fine tune. While the adjusted learning rate did knock the loss function out of saddle points, it overall had a negligible effect on the outcome of the model.
