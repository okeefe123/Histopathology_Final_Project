Histopathology Final Project
====
Introduction
----
In this project, the [Histopathologic Data Set](https://www.kaggle.com/c/histopathologic-cancer-detection) from Kaggle as a basis for exploring the capabilities of Convolutional Neural Networks. The primary goal in this project is to identify the model architecture which can most accurately predict whether an image of lung or colon tissue signifies the presence of cancerous cells.

Background
----

[Efrem will include background here]


Goals
----
Some goals we had in mind in our search for the most optimal modeling pipeline were as follows:

1. Explore: Given a list of predetermined CNN models (Mobilenet, Densenet, Alexnet, Resnet), which will give the best base results?
2. Refine: How can different approaches for the same model yield different approaches? In this step, we will identify whether it's more appropriate to use a single classification Neural Network for both lung and colon cancer images or create two separate models for these regions of the body.
3. Fine Tune: Once we've settled on a baseline model and establised a most viable architecture, how can we tweak hyperparameters such as learning rate and pretrained layer parameters increase the accuracy of predictions?
4. Apply Regularization: Taken into account in parallel with the fine-tuning phase, what are some additional tehniques which can prevent the chosen model from overfitting, and how effective are they at improving the validation accuracy of the model?

Explore: Model Selection
----
The first task in the journey to an optimal model is selecting the most appropriate pretrained model. For this part, we will ask the model to predict one of five classes from the image over five epochs: lung_n, lung_aca, lung_scc, colon_aca, or colon_n. While not entirely related to the question at hand, this will give us a good idea of the way a model behaves on medical images and provide the opportunity for more experience in interpreting Cross Entropy loss. The pretrained layers will be frozen for the moment, and the final 1000 class dense linear layer will instead be replaced with a 5 class linear layer for purposes of multi-class classification. The models tested are as follow:

a. [DenseNet](https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow) - Created to remedy the vanishing/exploding gradient problem of deep neural networks, this architecture seeks to retain information that might otherwise be lost to backpropogation when tweaking the parameters in the layers. It relies on concatenating the output of previous layers with future ones, of which presents the downside of being computationally expensive.

![DENSENET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/densenetloss.png)

While densenet may be suitable for many purposes, we see that in this case it's increasingly overfitting as the gap widens between the training and validation loss. Not shown is the accuracy, which relayed an impressive 0.8 for the training set but an abysmal 0.02 for validation. While this overfitting can be dealt with through regularization techniques, it may be best to survey the others before making that leap.

b. [AlexNet](https://en.wikipedia.org/wiki/AlexNet) - While not the first GPU-implementations of a CNN, this model was one of the first to publicize the technique and was proposed in one of the most cited machine learning papers on Google Scholar. It makes use of five convolutional layers connected via three interdispersed maxpool layers, with three fully connected dense layers before outputting 1 of 1000 classes. This is seen as a precursor to the other pretrained models discussed in this section. The idea in choosing this is that, while deep, is in a sweet spot in parameter count and thus has the capacity to give good results while not being too computationally expensive.

![ALEXNET](https://github.com/okeefe123/Histopathology_Final_Project/blob/main/figures/alexnetloss.png)

While the initial glance at the losses are very promising, further inspection shows similar lackluster results as DenseNet with respect to overfitting the training set. We do see the validation and training loss converging to a similar scale, which overtakes DenseNet as the best candidate so far.

c. Mobilenet

d. Resnet
