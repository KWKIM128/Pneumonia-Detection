# Pneumonia-Detection
<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221438788-9c94d249-08eb-46d6-b133-be284c0385b8.png" />
</p>
<p align="center"> Figure 1. Pneumonia Dataset
  
The dataset contains lung x-ray scans of patients with and without pneumonia [1]. The dataset contains two types of pneumonia, bacterial and viral, as shown in the figure above. However, the model developed only determines whether the x-ray scan has pneumonia or not as shown in figure 2.

<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221438834-5a3f01d6-0a8f-4f3c-aedc-1de66da0fa02.png" />
</p>
<p align="center"> Figure 2. X-Ray Scans With and Without Pneumonia


The developed model has an accuracy of 93.75 %, and the structure of the model is described in the figure below. Initially, 4 models developed by Kaushik et al (2020), were replicated, however, the replicated models showed signs of overfitting. Therefore, the third model was adapted to design a new model. The overall structure of the model was kept the same, however, the learning rate, loss function and activation function of the last dense layer were changed. In the paper, a categorical cross-entropy with softmax activation was used. This was changed to binary cross-entropy with sigmoid activation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221440420-8751f224-81c9-4383-8591-10e17ed414c6.png" />
</p>
<p align="center"> Figure 3. Model Structre

In addition, a decay learning rate was used, where the learning rate will decrease by a factor of 0.3 if the validation accuracy does not change for 5 epochs, see figure 4. To prevent overfitting, dropout layers and L1 regularization were introduced to the model. The overall performance of the adapted model has 1.44 % higher accuracy than the best-performing model in Kaushik et al (2020). The training loss and accuracy are shown in figures below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221439283-78b004a6-5350-44e2-9858-68c123fee3fe.png" />
</p>
<p align="center"> Figure 4. Training Loop

<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221439112-bebe7b0c-61ac-4e0e-8748-e9eff24a7043.png" />
</p>
<p align="center"> Figure 5. Training and Validation Loss

  
<p align="center">
  <img src="https://user-images.githubusercontent.com/115262940/221439090-9b8b6078-511f-4593-9f84-d8c55b11d5f9.png" />
</p>
<p align="center"> Figure 6. Training and Validation Accuracy


## References
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 

Sirish Kaushik, V., Nayyar, A., Kataria, G. and Jain, R., 2020. Pneumonia detection using convolutional neural networks (CNNs). In Proceedings of First International Conference on Computing, Communications, and Cyber-Security (IC4S 2019) (pp. 471-483). Springer Singapore.
