----Dataset----

To complete this task i use balanced split of EMNIST dataset.I choose this split because there is no imbalance among  classes what makes learning process
much easier.

During the exploratory data analysis i notice that images in this dataset are flipped upside down and then rotated 90 degrees clockwise.So i have to adress this
problem during inference step.

In the next step i have to get rid of unnecessary classes such as small latters and O, Q, I symbols.

In order to have my dataset larger to prevent overfitting and increase accuracy i also use data augmentation i use RandomAffine(degrees=10, translate=(0.1, 0.1), shear=(10, 10, 10, 10))
this helps the model to generalize better and gain better perfomance on the test dataset.

----Model----

I use ResNet34 model, because it is simple to use and to train it.I've also tried both ResNet18 and ResNet50 and understood that too shallow or too deep ResNet performs 
worse, whereas ResNet34 gained best perfomance among them.

----Training----

For training I choose batch_size = 512 and 25 epochs. I also devide TrainDataset into 2 parts, Train and Test, where Train is 0.95 of size of the original Train
after all manipulations.Test set is used after the model is trained to check it perfomance on the unseen data. I've done this step when i was training my model in
Google Colab just to insure, that my model does not overfit to the Train and Val data.But i dont do this in train.py script in order to gain more accuracy and information from TrainSet.

I use AdamW optimizer and ReduceLrOnPlateu schelduler with patience=3 and factor=0.1

Initial learning rate i get from the specific function, that evaluates it.

----Results---


results for the 25-th epoch: train Loss: 0.0840 Acc: 0.9658, val Loss: 0.1367 Acc: 0.9537

best val accuracy during training: Best val Acc: 0.954318

results for the test dataset: Test accuracy: 0.9515, Test F1 score: 0.9512

some pictures:

![alt text](https://i.imgur.com/BClqq0d.png)






![alt text](https://i.imgur.com/SQUxO8H.png)





----How To Use----

Download the repo

install requirements from requirements.txt

To train the model execute train.py BUT BE SURE THAT YOU HAVE CUDA INSTALLED BECAUSE THE SCRIPT USES CUDA TO TRAIN MODEL

To use trained model execute inference.py


----Additional Info----

email: jdackdack@gmail.com
link to github ipynb with training process https://github.com/StarLord202/CHI/blob/main/CHI.ipynb







