% Initialise
close all
clear
clc


% Matrix size
N = 400;

% Number of trials
numTrials = 100;

% Number of downdates
numDowndates = 10;

% Initialise time counters
timeMATLAB = 0;
timeMEX = 0;

% Initialise error counters
relErrorMATLAB = zeros(numTrials, 1);
absErrorMATLAB = zeros(numTrials, 1);
relErrorMEX = zeros(numTrials, 1);
absErrorMEX = zeros(numTrials, 1);

% Initialise the counters for successful updates
idx = 0;

% Scaling of the x vector. Small values lead to more successful downdates.
xScale = 1e-2;


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
    x = xScale* ...
        (rand(N, numDowndates) - 0.5).* ...
        (10.^((rand(N, numDowndates) - 0.5)));
    
    ticMATLAB = tic;
    RNewMATLAB = R;
    for k = 1 : 1 : numDowndates
        [RNewMATLAB, statusMATLAB] = cholupdate(RNewMATLAB, x(:, k), '-');
        if statusMATLAB == 1
            break
        end
    end
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
        
        exactPNew = R'*R;
        for k = 1 : 1 : numDowndates
            exactPNew = exactPNew - x(:, k)*x(:, k)';
        end
        
        absErrorMATLAB(i) = norm(RNewMATLAB'*RNewMATLAB - exactPNew, 'fro');
        relErrorMATLAB(i) = absErrorMATLAB(i)/norm(exactPNew, 'fro');
        
        absErrorMEX(i) = norm(RNewMEX'*RNewMEX - exactPNew, 'fro');
        relErrorMEX(i) = absErrorMEX(i)/norm(exactPNew, 'fro');
        
        fprintf('MEX absolute error: %.4e\n', absErrorMEX(i));
        fprintf('MEX relative error: %.4e\n', relErrorMEX(i));
        fprintf('\n');
        fprintf('MATLAB absolute error: %.4e\n', absErrorMATLAB(i));
        fprintf('MATLAB relative error: %.4e\n', relErrorMATLAB(i));
        fprintf('\n\n');
        
    elseif (statusMEX == -1) && (statusMATLAB == 1)
        fprintf('Could not downdate!\n\n');
    else
        error('Different MATLAB und MEX behaviour!');
    end
    
end


numSuccessful = idx;
absErrorMATLAB(numSuccessful + 1 : 1 : end) = [];
relErrorMATLAB(numSuccessful + 1 : 1 : end) = [];
absErrorMEX(numSuccessful + 1 : 1 : end) = [];
relErrorMEX(numSuccessful + 1 : 1 : end) = [];


fprintf('Total trials:        %12i\n', numTrials);
fprintf('Successful:          %12i\n', numSuccessful);
fprintf('Could not downdate:  %12i\n', numTrials - numSuccessful);
fprintf('\n');
fprintf('Mean MEX runtime:    %12.4f ms\n', timeMEX*1e3/numSuccessful);
fprintf('Mean MATLAB runtime: %12.4f ms\n', timeMATLAB*1e3/numSuccessful);
fprintf('Speedup:             %12.4f x\n', timeMATLAB/timeMEX);
fprintf('\n');
fprintf('Max MEX absolute error:  %.4e\n', max(absErrorMEX));
fprintf('Max MEX relative error:  %.4e\n', max(relErrorMEX));
fprintf('\n');
fprintf('Max MATLAB absolute error:  %.4e\n', max(absErrorMATLAB));
fprintf('Max MATLAB relative error:  %.4e\n', max(relErrorMATLAB));
