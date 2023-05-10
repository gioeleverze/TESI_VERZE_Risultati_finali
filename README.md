# TESI_VERZE
The repository contains the Jupiter notebook, the dataset, and the result of the thesis implementation. It analyzes two different datasets:


-[SKAB](https://github.com/waico/SKAB) 

-[EXATHLON](https://github.com/exathlonbenchmark/exathlon)


The next sections analyze the work done on the datasets and the structure of the repository.
## MODEL
### SKAB

The dataset is contained in the data folder. It is divided into three different parts classified according to the different types of anomaly that it contains.
All the models analyzed, except USAD and ELM-MI, are implemented in the Jupiter notebook, and to obtain the result, it is sufficient to run them.
```sh
run CONV_AE.ipynb
run LSTM.ipynb
run LSTM_AE.ipynb
run LSTM_VAE_reenc.ipynb
run LSTM_VAE_isof.ipynb
run LSTM_VAE_mse.ipynb
run LSTM_VAE_knn.ipynb
``` 
Moreover, all models are yet trained and saved in the relative model folder.

Concerning USAD implementation, it is sufficient to run the Jupiter notebook contained in the relative folder.
```sh
cd USAD
run USAD_SKAB.ipynb
``` 

For ELM-MI, the execution is a bit more complicated. 
First, it has to generate the dataset to process with Maltlab. 

```sh
cd ELM-MI
``` 
Then it is possible to execute the Matlab script by running the file ELMMI_DKS.m with the Matlab tool. The results obtained are saved in the Result folder. To have a complete execution of all the files of the dataset are necessary three execution of the script ELMMI_DKS.m. For each execution, there is a little change to do in the script.

First execution:
```sh
line 7) d=dir('./../data/valve2/*.csv');  
line 8) %d=dir('./../data/other/*.csv');  
line 9) %d=dir('./../data/valve1/*.csv');  
``` 
Second execution:
```sh
line 7) %d=dir('./../data/valve2/*.csv');  
line 8) d=dir('./../data/other/*.csv');  
line 9) %d=dir('./../data/valve1/*.csv');  
``` 
Third execution:
```sh
line 7) %d=dir('./../data/valve2/*.csv');  
line 8) %d=dir('./../data/other/*.csv');  
line 9) d=dir('./../data/valve1/*.csv');  
``` 
Once executed it is necessary to evaluate the score:

```sh
cd Result
run matlab_SKAB_result.ipynb
``` 


### EXATHLON

The dataset is contained in two different folders, the training data are contained in the DATA_SPLITTED folder, and the testing data are in the folder OUTPUT_ROOT.
All the models analyzed, except USAD and ELM-MI, are implemented in the Jupiter notebook, and to obtain the result, it is sufficient to run them.
```sh
run CONV_AE.ipynb
run Dense_AE.ipynb
run LSTM.ipynb
run VAE_REENC.ipynb
run VAE_REENC_alpha_beta.ipynb
run VAE_isof_knn.ipynb
``` 
Moreover, all models are yet trained and saved in the relative model folder.

Concerning USAD implementation, it is sufficient to run the Jupiter notebook contained in the relative folder.
```sh
cd USAD
run USAD.ipynb
``` 

For ELM-MI, the execution is a bit more complicated. 
First, it has to generate the dataset to process with Maltlab. 

```sh
cd ELM-MI
cd data
run Dataset_to_matlab.ipynb
``` 

Then, after returning to the ELM-MI folder, it is possible to execute the Matlab script by running the file ELMMI_DKS.m with the Matlab tool. The results obtained are saved in the Result folder. To evaluate the result is necessary:

```sh
cd Result
run Score_evaluation.ipynb
``` 

The results folder contains all the results obtained with different models.
The score_analisys folder contains an analysis of the validation score used to compute the threshold.

```sh
cd score_analisi
run Exathlon_validation_box_plot.ipynb
``` 
## RESULT
The folder contains, for both datasets:
GENERATE_RESULT.ipynb : to evaluate the score end generating the results by using the threshold
Distribution.ipynb : to generate the anomaly score distribution and the plots
VAL and SCORE : which contains the anomaly score of validation and testing
GRAFICI : which contains the ROC Curve graph and the anomaly score distribution plots


## SCORE
The folder contains, for both datasets and all the models implemented, the anomaly score corresponding to the training, validation, and testing.




