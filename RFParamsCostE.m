function [z,Out] = RFParamsCostE(x,Model)

Features = Model.Feature;

Labels = Model.Label;

K = Model.K;
CVI = Model.CVI;

%% Map the x into FR parameyters
Options = MapVar2Params(x,Features);

%% Create the Tree template
t = templateTree('MaxNumSplits' , Options.MaxNumSplits, ...
    'PredictorSelection',Options.PredictorSelection,...
    'NumVariablesToSample',Options.NumVariablesToSample,...
    'MergeLeaves', Options.MergeLeaves,...
    'MinLeafSize', Options.MinLeafSize,...
    'SplitCriterion' ,Options.SplitCriterion);
    
AccuracyTrain = zeros(1,K);
AccuracyValid = zeros(1,K);
%% Main Loop Train
for i = 1:K

    TrainF = Features(CVI~=i,:);
    TrainL = Labels(CVI~=i);
 
    Classify = fitcensemble(TrainF,TrainL,'Learners',t,...
            'NumLearningCycles',Options.NumLearningCycles,...
            'Method' , 'Bag' );
 
    ClassT = predict(Classify,TrainF);
    CMT = confusionmat(TrainL,ClassT);
    AccuracyTrain(i) = sum(diag(CMT))/sum(CMT(:));

    ValidF = Features(CVI==i,:);
    ValidL = Labels(CVI==i);

    ClassV = predict(Classify,ValidF);
    CMV = confusionmat(ValidL,ClassV);
    AccuracyValid(i) = sum(diag(CMV))/sum(CMV(:));

    Cl(i).Classify = Classify;
end
AccuracyValid1 = mean([AccuracyTrain;AccuracyValid]);
[~,idxm] = max(AccuracyValid1);
Classify = Cl(idxm).Classify;
% Main Cost Value
z = 1 - mean(AccuracyValid1);

Out.z = z;
Out.Classify = Classify;
Out.AccuracyValid = AccuracyValid;
Out.AccuracyTrain = AccuracyTrain;
Out.Data = Model;
Out.Params = Options;
end


function  Options = MapVar2Params(x,Features)
[nObs,nVar] = size(Features);
Options.MaxNumSplits = 1 + round((nObs-1)*x(1));
Options.MinLeafSize = max(1,round(10*x(2)));
Options.NumLearningCycles = 100 + round(100*x(3));
Options.NumVariablesToSample = 2 + round(5*nVar*x(4));
SplitCriterion = min(max(1,round(3*x(5))),3);
PredictorSelection = min(max(1,round(3*x(6))),3);

if x(7) > 0.5
    Options.MergeLeaves = 'on';
else
    Options.MergeLeaves = 'off';
end

switch PredictorSelection
    case 1
        PredictorSelections = 'allsplits';
    case 2
        PredictorSelections =  'interaction-curvature';
    case 3
        PredictorSelections =  'curvature' ;
end
Options.PredictorSelection = PredictorSelections;

switch SplitCriterion
    case 1
        SplitCriterions = 'gdi';
    case 2
        SplitCriterions =  'twoing';
    case 3
        SplitCriterions =  'deviance' ;
end
Options.SplitCriterion = SplitCriterions;
end
