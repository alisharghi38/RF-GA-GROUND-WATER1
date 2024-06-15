function Results = OptimizeRFusingGA(TrainModel,Params)
disp('Starting RF-GA Ali_Sharghi ...')
%% Problem Definition



if contains(TrainModel.Function,'Ens')
    CostFunction = @(x)RFParamsCostE(x,TrainModel);
    nVar = 7;
else
    CostFunction = @(x)RFParamsCostB(x,TrainModel);
    nVar = 5 ;   % Number of Decision Variable
end
VarSize = [1,nVar];

VarMin = 0; % Lower Bound of Variable
VarMax = 1; % Upper Bound of Variable


%% GA Parameters

MaxIt = Params.MaxIt;      % Maximum Number of Iterations

nPop = Params.nPop;        % Population Size

pc = 0.8;                 % Crossover Percentage
nc = 2*round(pc*nPop/2);  % Number of Offsprings (Parnets)
%gamma = 0.05;

pm = 0.3;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu = 0.02;         % Mutation Rate
%% Initialization(p#0)
empty_individual.Position = [];
empty_individual.Cost = [];

pop = repmat(empty_individual,nPop,1);
tic
for i = 1:nPop
    
    % Initialize Position
    pop(i).Position = unifrnd(VarMin,VarMax,VarSize);
    
    % Evaluation
    pop(i).Cost = CostFunction(pop(i).Position);
    
end

% Sort Population
Costs = [pop.Cost];
[Costs, SortOrder] = sort(Costs);
pop = pop(SortOrder);

% Store Best Solution
BestSol = pop(1);

% Array to Hold [Best Worst Mean] Cost Values
BestCost = zeros(MaxIt,1);

% Store Cost
WorstC = Costs(end);

% Array to Hold Number of Function Evaluations
nfe = zeros(MaxIt,1);

NFE = nPop;
%% Main Loop

for it = 1:MaxIt
    
    % Calculate Selection Probabilities
    P = exp(-8*Costs/WorstC);
    P = P/sum(P);
    
    % Crossover
    popc = repmat(empty_individual,nc/2,2);
    for k = 1:nc/2
        
        % Select Parents Indices(p#1)
        i1 = RouletteWheelSelection(P);
        i2 = RouletteWheelSelection(P);
        
        
        % Select Parents
        p1 = pop(i1);
        p2 = pop(i2);
        
        % Apply Crossover(p#2)
        [popc(k,1).Position ,popc(k,2).Position] = ...
            Crossover(p1.Position,p2.Position,VarMin,VarMax);
        
        % Evaluate Offsprings
        popc(k,1).Cost = CostFunction(popc(k,1).Position);
        popc(k,2).Cost = CostFunction(popc(k,2).Position);
        
    end
    popc = popc(:);
    
    
    % Mutation(p#3)
    popm = repmat(empty_individual,nm,1);
    for k = 1:nm
        
        % Select Parent
        i = randi([1 nPop]);
        p = pop(i);
        
        % Apply Mutation
        popm(k).Position = Mutate(p.Position,mu,VarMin,VarMax);
        
        % Evaluate Mutant
        popm(k).Cost = CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population and Apply Elitism(p#4)
    pop= [pop
        popc
        popm];
    
    % Sort Population
    Costs = [pop.Cost];
    [Costs, SortOrder] = sort(Costs);
    pop = pop(SortOrder);
    
    % Update Worst Cost and Mean Cost
    WorstC = max(WorstC,pop(end).Cost);
    
    
    % Truncation(p#5)
    pop = pop(1:nPop);
    Costs = Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol = pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it) = BestSol.Cost;
    
    % Store NFE
    NFE = NFE  + nm + nc;
    nfe(it) = NFE;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': NFE = ' num2str(nfe(it)) ', Best Cost = ' num2str(BestCost(it))]);
    
end

%% Results
toc

t  = toc;
figure;

plot(1:MaxIt,BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Cost');
title('Trend of Optimizing RF Using GA');



[~,Out] = CostFunction(BestSol.Position);
Results.BestCost = BestCost;
Results.BestSol = BestSol;

Results.Time = t;
Results.Out = Out;

end



function [y1 ,y2]=Crossover(x1,x2,VarMin,VarMax)

    alpha=unifrnd(VarMin,VarMax,size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    %y1=max(y1,VarMin);
    %y1=min(y1,VarMax);
    
    %y2=max(y2,VarMin);
    %y2=min(y2,VarMax);

end

function y=Mutate(x,mu,VarMin,VarMax)

    nVar=numel(x);
    
    nmu=ceil(mu*nVar);
    
    j=randsample(nVar,nmu);
    
    sigma=0.2*(VarMax-VarMin);
    
    y=x;
    y(j)=x(j)+sigma*randn(size(j));
    
    y=max(y,VarMin);
    y=min(y,VarMax);

end

function i=RouletteWheelSelection(P)

    r=rand;
    
    c=cumsum(P);
    
    i=find(r<=c,1,'first');

end