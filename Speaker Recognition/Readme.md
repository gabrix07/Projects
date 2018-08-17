# Speaker Recognition

The aim of this project is to create system that should recognize the speaker by using voice data. This system could classify different speakers through artificial neural network. The neural network was trained by unsupervised learning, in addition to simulate
an analogous neural network, which was formed by memristors, resistances and electric capacitors. This classification method was analysed and finally compared to other methods commonly used in
speaker recognition:

* Vector quantization - 32 Vectors
* Gaussian Mixture Models - 4 Gaussian
* Neural Network (superviced learning) - 64 hidden neurons

The programs were written in **Matlab**

---

## Folder Information 

* **__Classifier__**: In this folder the different classification models are stored.
* **__GUI__**: This Folder contains the graphical user interface. In this way it is possible to train and test your own experiments. 
For this it is necessary to have a **microphone** and **Matlab**. All you have to do is compile the files _Evaluation.m_ and _Training.m_.
* **_MFCC_**: In this folder contains the Mel-frequency cepstral coefficients (MFCC) of the speakers used in this project. They were saved as _.mat_.
* **Speech Data**: Here you can find the audio database used for training of the models and their evaluation.

## Results

| Classification Method | Number of Speakers | Speaker Recognition |
| -- |-- | -- |
| VQ |5 | 87% |
| GMM |5 | 88% |
| ANN (superviced) |5 | 89% |
| ANN (unsuperviced)|5 | 85% |
| VQ |10 | 79% |"
| GMM |10 | 79% |
| ANN (superviced) |10 | 83% |
| ANN (unsuperviced)|10 | 70% |

Test:
![alt text](https://github.com/gabrix07/Projects/tree/master/Speaker%20Recognition/Hebbvsback.png "Logo Title Text 1")
