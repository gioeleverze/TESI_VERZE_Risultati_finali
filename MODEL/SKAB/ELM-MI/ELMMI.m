%% 
clear; close all; clc;
rng(0); % fix the random seed


%% 
%d=dir('./../data/valve2/*.csv');  
%d=dir('./../data/other/*.csv');  
d=dir('./../data/valve1/*.csv');  
iterator=0;
for file=1:length(d)
   % Open the FIle
   fid=0;

   fid=readtable(strcat(strcat(d(file).folder,'\'),d(file).name));
   M=fid(400:size(fid),2:9);
   save('test.mat','M');
   M=fid(1:400,2:9);
   save('train.mat','M');
   M=fid(400:size(fid),10);
   save('anomaly.mat','M');
   % load the dataset 
    Data1=load('./train');
    train=table2array(Data1.M)';
    Data2=load('./test');
    test=table2array(Data2.M)';
    Data3=load('./anomaly');
    anomaly=table2array(Data3.M);
    
    
    
    figure(1)
    subplot(2,1,1)

    plot(train(4,:), 'r-');

    subplot(2,1,2)


    
measuredMI_ELMMImulti=algorithm(train_,test_);   
figure(1)
subplot(2,1,1)

plot(anomaly, 'r-');

title('Demo SKAB '); 
subplot(2,1,2)

plot(measuredMI_ELMMImulti, 'b-','LineWidth',1,'MarkerSize',8);


ylabel('Anomaly Score');
title('Demo Test Result '); 



stampa=measuredMI_ELMMImulti';
stringa='./score';
stringa=strcat(stringa,d(file).name);
writematrix(stampa,stringa)

iterator=iterator+1;



end




%% 
%% 
