Finding relations between p and q of contact_binaries using ML methods



In this project, we have trained a Multi-layer Perceptron (MLP) regressor on periods of 371 W UMa contact binaries to obtain their mass ratio. In order to find the confidence interval, we have used a bootstrapping approach in which we train MLP models separately in 500 steps. At each step the model is fitted to 90% of the data that is randomly selected.

You can use our model to estimate the mass ratio of a W Uma contact binary using only its period. In  order to do that, go inside the folder "codes" in your terminal. Type:

python W_UMa_q_prediction.py 0.282

Where the number 0.282 is the value of period that can be different for any other objects.

If you want  to recreate the Figure 10 of the paper, you can type the command below in your terminal:

python W_UMa_q_prediction.py 0.282 plot

Please note that you need the following python packages to run the code:

numpy
pandas
matplotlib
joblib
tqdm
scipy

Please let us know if you have any questions and cite our paper if you use the content in this GitHub repository.