function [TrainData,TestData] = LoadData(name)


%% Load Data
data = importdata(['DataSets/',name]);
if isstruct(data)
    data = data.data;
end
%% Manage Data

%y_silakhur
Y_Silakur = data(:,end);
%x_silakhur
X_Silakur = data(:,1:end-1);

% Normalization
%X_Silakur = Normalaization(X_Silakur);
%numbers are the same eatch round if rng(0)
rng(0)
% Divide TrianData And TestData
%percentage of train data
pT = 80;
%nFeature= number of conciderd before_month for pizometer and precipitation 
%nSamples= number of recorded data for pizometers or precipitation
[nSamples,nbefore_month] = size(X_Silakur);
nTrain = round((pT/100)*nSamples);
%randoming process for classification
%R = randperm(nSamples);
%indTrain = R(1:nTrain);
%indTest = R(nTrain+1:end);

X_SilakurTR = X_Silakur(1:nTrain,1:nbefore_month);
TrainData.X_SilakurTR= X_SilakurTR;
TrainData.X_Silakur= X_Silakur;
TrainData.Y_Silakur= Y_Silakur;
Y_SilakurTR = Y_Silakur(1:nTrain);
TrainData.Y_SilakurTR= Y_SilakurTR;
TrainData.nbefore_month = nbefore_month;

X_SilakurTST = X_Silakur(nTrain+1:end,1:nbefore_month);
TestData.X_SilakurTST=X_SilakurTST;
Y_SilakurTST = Y_Silakur(nTrain+1:end,:);
TestData.Y_SilakurTST=Y_SilakurTST;

 %Apply K-Fold Cross Validation
  K = 10;
  vectorsample=X_SilakurTR(:,1);
  NT = numel(vectorsample);
  CVI = crossvalind('kfold',NT,K);
  TrainData.CVI = CVI;
  TrainData.K = K;

end

% function X = Normalaization(X,LB,UB)
% 
% Min = min(X);
% Max = max(X);
% if nargin<2
%     LB = -1;
%     UB = 1;
% end
% 
% for i = 1:numel(Min)
%     X(:,i) = (X(:,i) -Min(i))/(Max(i) - Min(i)) ;
% end
% X = (UB - LB)*X + LB;
% 
% 
% end
