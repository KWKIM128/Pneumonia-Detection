# Pneumonia-Detection

Figure 1. Pneumonia Dataset
The dataset contains lung x-ray scans of patients with and without pneumonia [1]. The dataset contains two types of pneumonia, bacterial and viral, as shown in the figure above. However, the model developed only determines whether the x-ray scan has pneumonia or not as shown in figure 2.


Figure 2. Xiray scans with and without Pneumonia


The developed model has an accuracy of 93.75 %, and the structure of the model is described in the figure below. Initially, 4 models developed by Kaushik et al (2020), were replicated, however, the replicated models showed signs of overfitting. Therefore, the third model was adapted to design a new model. The overall structure of the model was kept the same, however, the learning rate, loss function and activation function of the last dense layer were changed. In the paper, a categorical cross-entropy with softmax activation was used. This was changed to binary cross-entropy with sigmoid activation.

Figure 3. Model structre

In addition, a decay learning rate was used, where the learning rate will decrease by a factor of 0.3 if the validation accuracy does not change for 5 epochs. To prevent overfitting, dropout layers and L1 regularization were introduced to the model. The overall performance of the adapted model has 1.44 % higher accuracy than the best-performing model in Kaushik et al (2020). The training loss and accuracy are shown in figures below

