%MLE_to_BF.m
%
%This programme takes the likelihood or log-likelihoods for 2 models
%computed using Maximum Likelihood estimation and converts these values
%into Bayes factors comparing those models.

clear variables;
clc;

N=input('Number of data points in data being fitted by the models= ');
raw_v_log=input('Choose whether to use raw likelihoods (1) or log-likelihoods (2)= ');
disp(' ');
disp('Model 1');
if raw_v_log==1
    L1=input('Likelihood of MLE estimate under this model= ');
    K1=input('Number of free parameters used in this model= ');
    disp('Model 2');
    L2=input('Likelihood of MLE estimate under this model= ');
    K2=input('Number of free parameters used in this model= ');
    BIC1=-2*log(L1)+K1*log(N);
    BIC2=-2*log(L2)+K2*log(N);
elseif raw_v_log==2
    LL1=input('Log Likelihood of MLE estimate under this model= ');
    K1=input('Number of free parameters used in this model= ');
    disp('Model 2');
    LL2=input('Log Likelihood of MLE estimate under this model= ');
    K2=input('Number of free parameters used in this model= ');
    BIC1=-2*LL1+K1*log(N);
    BIC2=-2*LL2+K2*log(N);
end;

disp(['BIC for model 1= ' num2str(BIC1)]);
disp(['BIC for model 2= ' num2str(BIC2)]);
BF_1vs2= exp(-0.5*BIC1)./exp(-0.5*BIC2);
disp(['BF M1 vs M2= ' num2str(BF_1vs2)]);
disp('Using Jeffrey''s description of the evidential value of Bayes factors, this represents:-');
if BF_1vs2 < 1 %#ok<BDSCI>
    bestmodel=2;
else
    bestmodel=1;
end;
if bestmodel==1
    BF=BF_1vs2;
elseif bestmodel==2
    BF=1./BF_1vs2;
end;
if BF > 100
    evid_label='decisive ';
elseif BF>30 && BF<=100
    evid_label='very strong ';
elseif BF>10 && BF<=30
    evid_label='strong ';
elseif BF>3 && BF<=10
    evid_label='substantial ';
elseif BF>1 && BF<=3
    evid_label='anecdotal ';
elseif BF==1
    evid_label='no ';
end;

disp([evid_label 'evidence in favour of model ' num2str(bestmodel)]);