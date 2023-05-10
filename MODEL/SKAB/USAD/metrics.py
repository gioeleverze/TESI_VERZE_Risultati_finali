

import pandas as pd
import numpy as np


def evalu(true, prediction,metric='nab',window_time=None):

  
  #binary evaluation with parameter TP,TN,FN,FP
  def binary(true,prediction):

    def single_binary(true,prediction):
      true_= true==1
      prediction_ = prediction==1
      TRUE=true.sum()
      #applico il .sum in quanto true non è formato da un solo valore ma
      #da un insieme di valori in quanto i nel for itera su 33 elementi
      TP = (true_ & prediction_).sum()     #truepositive       outlier corretto
      TN = (~true_ & ~prediction_).sum()   #truenegative       no outlier corretto
      FP = (~true_ & prediction_).sum()    #falsepositive      outlier ma non lo è
      FN = (true_ & ~prediction_).sum()    #falsenegative      non outlier ma lo è
      return TP,TN,FP,FN,TRUE

    TP,TN,FP,FN,TRUE=0,0,0,0,0
    #if type(prediction) != type(list()):
    if len(prediction)==1:
      TP_,TN_,FP_,FN_,TRUE_=single_binary(np.array(true),np.array(prediction))  
      TP,TN,FP,FN,TRUE=TP+TP_, TN+TN_, FP+FP_, FN+FN_,TRUE+TRUE_
      #print("different len")  
    else:
      for i in range(len(true)):

        TP_,TN_,FP_,FN_,TRUE_=single_binary(true[i],prediction[i])
        TP,TN,FP,FN,TRUE=TP+TP_, TN+TN_, FP+FP_, FN+FN_,TRUE+TRUE_

    
    PREC=TP / (TP + FP)
    REC = TP/ TRUE
    ACC = (TP + TN)/(TP + TN + FP + FN)
    f1=2 * PREC * REC/(PREC + REC)
    #f1 = round(TP/(TP+(FN+FP)/2), 2)

    print(f'Total value: {TP+TN+FP+FN}')
    print(f'TRUE POSITIVE: {TP}')
    print(f'TRUE NEGATIVE: {TN}')
    print(f'FALSE POSITIVE: {FP}')
    print(f'FALSE NEGATIVE: {FN}')
    print(f'PRECISION: {PREC}')           #frazione di casi identificati come positivi che sono correttamente positivi.
    print(f'RECALL: {REC}')               #frazione di positivi che sono identificati dal modello come positivi
    print(f'ACCURANCY: {ACC}')            #valori esatti rispetto al totale dei valori
    print(f'False Alarm Rate {round(FP/(FP+TN)*100,2)} %' )
    print(f'Missing Alarm Rate {round(FN/(FN+TP)*100,2)} %')
    print(f'F1 metric {f1}')
    return f1

  #calculation of average delay using changepoint value
  def average_delay(true_cp, prediction):

    def single_average_delay(true_cp,prediction):
      missing,detection=0,[]
      for couple in true_cp:

        t1=couple[0]
        t2=couple[1]

        if prediction[t1:t2].sum()==0:
          missing+=1
        else:
          detection.append(prediction[prediction ==1][t1:t2].index[0]-t1)
      return missing,detection

    missing, detection=0,[]
    if type(prediction) != type(list()):
      missing, detectHistory = single_average_delay(true_cp, prediction)
    else:
      for i in range(len(prediction)):
        missing_,detection_=single_average_delay(true_cp[i],prediction[i])
        missing,detection=missing+missing_,detection+detection_

    avg_delay=pd.Series(detection).mean()
    print(f'Average delay {avg_delay}')
    print(f'Missing ChangePoint {missing}')
    return avg_delay

  #numenta anomaly benchmark




  #controllo se i valori passati sono changepoint o outlier (per binary ho outlier)
  if not metric=='binary':
    #considero solo i valori =1 nei punti che sono changepoint
    if type(true) != type(list()):
        true_items = true[true==1].index       #index(datetime) dei valori uguali a 1
    else:
        true_items = [true[i][true[i]==1].index for i in range(len(true))]

 
    def single_detecting_boundaries(true, window_time, true_items):
        detection=[]
        
        td = pd.Timedelta(window_time)
        
        #if window_time is not None else pd.Timedelta((true.index[-1]-true.index[0])/len(true_items))  
        for val in true_items:
            detection.append([val, val + td])
        return detection

    if type(true) != type(list()):
        detection = single_detecting_boundaries(true=true,window_time=window_time, true_items=true_items)
    else:
        detection=[]
        for i in range(len(true)):
            detection.append(single_detecting_boundaries(true=true[i], window_time=window_time, true_items=true_items[i]))



  if metric=='nab':
    print("nab")
  elif metric=='avg':
    average_delay(detection,prediction)
  elif metric=='binary':
    binary(true,prediction)