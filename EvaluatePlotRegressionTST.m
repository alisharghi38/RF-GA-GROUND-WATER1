function ResultsE = EvaluatePlotRegressionTST(TrainModel,Results)
Function = TrainModel.Function;

Classify = Results.Out.Classify;
X_SilakurTST = TrainModel.X_SilakurTST;
Y_SilakurTST = TrainModel.Y_SilakurTST;
[predictedTST,Score] = predict(Classify,X_SilakurTST);
actualTST =Y_SilakurTST;
MSE=(sum(predictedTST-actualTST).^2)/length(actualTST);
RMSE=MSE^0.5;
 %Store Results
 ResultsE.Name = 'TSTpizo_pizo.mat';
 ResultsE.RMSE = RMSE;
 ResultsE.MSE = MSE;
 ResultsE.Error = actualTST - predictedTST;
figure;
subplot(2,2,[1 2]),
t = 1:numel(actualTST);
plot(t,actualTST,t,predictedTST,'--','LineWidth',1.5)
legend('Y-Actual','Y-Predict');
title(['Curve Fitting: for ',' TEST'])
subplot(2,2,[3 4]),hist(actualTST-predictedTST,20)
title(['Hist Error, RMSE = ',num2str(RMSE)]);
figure, plotregression(actualTST,predictedTST,'Regression');
 xlabel(['actual, Regression for ',' TEST'])
 ylabel(['predicted, Regression for ',' TESR'])
 disp(Results);

end