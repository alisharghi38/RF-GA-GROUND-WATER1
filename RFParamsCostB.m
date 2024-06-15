function [z,Out] = RFParamsCostB(x,Model)

X_SilakurTR = Model.X_SilakurTR;

Y_SilakurTR = Model.Y_SilakurTR;
 K = Model.K;
 CVI = Model.CVI;

%% Map the x into FR parameyters
Options = MapVar2Params(x,X_SilakurTR);

RMSEAccuracyTrain = zeros(1,K);
RMSEAccuracyValid = zeros(1,K);
%tic
for i = 1:K
   
    TrainF = X_SilakurTR(CVI~=i,:);
    TrainL = Y_SilakurTR(CVI~=i);

    Classify = TreeBagger(Options.nTrees,TrainF,TrainL,...
        'NumPredictorsToSample',Options.NumPredictorsToSample,...
        'MaxNumSplits',Options.MaxNumSplits,...
        'MinLeafSize',Options.MinLeafSize,...
        'PredictorSelection',Options.PredictorSelection,...
        'MergeLeaves',Options.MergeLeaves , ...
        'OOBPrediction','on',...
        'Method','regression');

    ClassT = predict(Classify,TrainF);

    %ClassT = str2double(cell2mat(ClassT));
    % Root Mean Squared Error
    RMSETRAIN = (mean((ClassT -TrainL).^2))^0.5;
    RMSEAccuracyTrain(i)= RMSETRAIN;
    ValidF = X_SilakurTR(CVI==i,:);
    ValidL = Y_SilakurTR(CVI==i);

    ClassV = predict(Classify,ValidF);
    %ClassV = str2num(cell2mat(ClassV));
    %add RMSE ok
    
    
% Root Mean Squared Error
RMSEVALIDATION = (mean((ClassV -ValidL).^2))^0.5;

    RMSEAccuracyValid(i) = RMSEVALIDATION;

    Cl(i).Classify = Classify;

end
%toc
%timetree=toc;
% disp(Options)
%AccuracyValid1 = min(RMSEAccuracyValid);
AccuracyValid1 = mean([RMSEAccuracyValid;RMSEAccuracyTrain]);
%Final_Best_RMSE=FBR
FBR=min(AccuracyValid1);
[~,id_best_TOTAL]=min(AccuracyValid1);
Classify = Cl(id_best_TOTAL).Classify;
%re write z for RMSE
z = FBR;
Out.z = z;
Out.Classify = Classify;
Out.FBR = FBR;
Out.RMSEAccuracyTrain = RMSEAccuracyTrain(id_best_TOTAL);
Out.RMSEAccuracyValid = RMSEAccuracyValid(id_best_TOTAL);
Out.Data = Model;
Out.Params = Options;
end

function Options = MapVar2Params(x,X_Silakur)
[nObs,nPredictors] = size(X_Silakur);
% Set Hyper-Parameters
Options.nTrees = 300 + round(400*x(1));
Options.MaxNumSplits = 1 + round((nObs-1)*x(2));
Options.MinLeafSize = 5+round(5*x(3));
Options.NumPredictorsToSample = 2+round((nPredictors-2)*x(4));
A=round(3*x(5));
PredictorSelection = max(A);
switch PredictorSelection
    case 1
        PredictorSelections = 'allsplits';
    case 2
        PredictorSelections =  'interaction-curvature';
    otherwise
        PredictorSelections =  'curvature' ;
end
Options.PredictorSelection = PredictorSelections;


Options.MergeLeaves = 'off';

end


