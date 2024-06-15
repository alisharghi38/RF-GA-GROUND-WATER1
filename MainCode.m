clc
clear
close all
%% Optimization of RF Parameters Using GA

%% load Data
% prumpt='number of forcast years = ';
% years = input (prumpt); % Number of years to forecast
% month=years*12;
NameData = 'MAINDATA.mat';
[TrainModel,TestData] = LoadData(NameData);
%% Function Selection 
Functions = {'Ensemble','Bagger'};
Function = Functions{2};
TrainModel.Function = Function;
TestData.Function = Function;

%% MA Sets Params
Params.nPop =30;
Params.MaxIt =10;
K=10;
%% Optimize RF using GA
Results = RF_GA_Ali_Sharghi(TrainModel,Params);
%% Prediction AND Evaluation Confusion Matrix & ROC
ResultsTrain = EvaluatePlotRegression(TrainModel,Results);
ResultsTest = EvaluatePlotRegressionTST(TestData,Results);
%forcast_2050
% ResultsForcast = forcast_2050(Results,TrainModel);

%% Displey
disp('********* Results **********');
disp(ResultsTrain);
disp(ResultsTest);
disp('******** RF Params *********');
disp(Results.Out.Params)
disp('********* Final Results *******')


