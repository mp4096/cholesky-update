% Initialise
close all
clear
clc


% Matrix size
N = 400;

% Number of trials
numTrials = 100;

% Initialise time counters
timeMATLAB = 0;
timeMEX = 0;

% Initialise error counters
relError = zeros(numTrials, 1);
absError = zeros(numTrials, 1);

% Initialise the counters for successful updates
idx = 0;

% Scaling of the x vector. Small values lead to more successful downdates.
xScale = 1;


for i = 1 : 1 : numTrials
    
    % =====================================================================
    % Create a positive definite diagonal matrix
    % =====================================================================
    % Create random positive eigenvalues
    positiveEigs = 10.^sum(rand(N, 3)*diag([10, 1, 0.1]), 2);
    Lambda = abs(diag(positiveEigs));
    % Get a random orthonormal matrix
    [Q, ~] = qr(rand(N));
    % Get a random s.p.d. matrix
    P = Q'*Lambda*Q;
    % =====================================================================
    
    % Perform Cholesky decomposition
    R = chol(P, 'upper');
    
    % Get a random downdate vector
    x = xScale*(rand(N, 1) - 0.5).*(10.^((rand(N, 1) - 0.5)));
    
    ticMATLAB = tic;
    [RNewMATLAB, statusMATLAB] = cholupdate(R, x, '-');
    if statusMATLAB == 0
        timeMATLAB = timeMATLAB + toc(ticMATLAB);
    end
    
    ticMEX = tic;
    [RNewMEX, statusMEX] = CholeskyDowndateReal(R, x);
    if statusMEX == 0
        timeMEX = timeMEX + toc(ticMEX);
    end
    
    if (statusMEX == 0) && (statusMATLAB == 0)
        idx = idx + 1;
        
        absError(idx) = norm(RNewMATLAB - RNewMEX, 'fro');
        relError(idx) = absError(idx)/norm(RNewMATLAB, 'fro');
        
        fprintf('Absolute error: %.4e\n', absError(idx));
        fprintf('Relative error: %.4e\n', relError(idx));
        fprintf('\n');
    elseif (statusMEX == -1) && (statusMATLAB == 1)
        fprintf('Could not downdate!\n\n');
    else
        error('Different MATLAB und MEX behaviour!');
    end
    
end


numSuccessful = idx;
absError(numSuccessful + 1 : 1 : end) = [];
relError(numSuccessful + 1 : 1 : end) = [];


fprintf('Total trials:        %12i\n', numTrials);
fprintf('Successful:          %12i\n', numSuccessful);
fprintf('Could not downdate:  %12i\n', numTrials - numSuccessful);
fprintf('\n');
fprintf('Mean MEX runtime:    %12.4f ms\n', timeMEX*1e3/numSuccessful);
fprintf('Mean MATLAB runtime: %12.4f ms\n', timeMATLAB*1e3/numSuccessful);
fprintf('Speedup:             %12.4f x\n', timeMATLAB/timeMEX);
fprintf('\n');
fprintf('Mean absolute error: %.4e\n', mean(absError));
fprintf('Mean relative error: %.4e\n', mean(relError));
