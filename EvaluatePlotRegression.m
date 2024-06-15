function ResultsE = EvaluatePlotRegression(TrainModel,Results)
Function = TrainModel.Function;

Classify = Results.Out.Classify;
X_SilakurTR = TrainModel.X_SilakurTR;
Y_SilakurTR = TrainModel.Y_SilakurTR;
[predictedTR,Score] = predict(Classify,X_SilakurTR);
actualTR =Y_SilakurTR;
MSE=(sum(predictedTR-Y_SilakurTR).^2)/length(Y_SilakurTR);
RMSE=MSE^0.5;
 %Store Results
 ResultsE.Name = 'TRAINpizo_pizo.mat';
 ResultsE.RMSE = RMSE;
 ResultsE.MSE = MSE;
 ResultsE.Error = actualTR - predictedTR;
figure;
subplot(2,2,[1 2]),
t = 1:numel(actualTR);
plot(t,actualTR,t,predictedTR,'--','LineWidth',1.5)
legend('Y-Actual','Y-Predict');
title(['Curve Fitting: for ','TRAIN'])
subplot(2,2,[3 4]),hist(actualTR-predictedTR,20)
title(['Hist Error, RMSE = ',num2str(RMSE)]);
figure, plotregression(actualTR,predictedTR,'Regression');
 xlabel(['actual, Regression for ',' TRAIN'])
 ylabel(['predicted, Regression for ',' TRAIN'])
 disp(Results);

end

