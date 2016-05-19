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
absErrorMEX = zeros(numTrials, 1);
relErrorMEX = zeros(numTrials, 1);
absErrorMATLAB = zeros(numTrials, 1);
relErrorMATLAB = zeros(numTrials, 1);


for i = 1 : 1 : numTrials
    
    % =====================================================================
    % Create a positive definite diagonal matrix
    % =====================================================================
    % Create random positive eigenvalues
    positiveEigs = 10.^sum(rand(N, 3)*diag([1, 1, 0.1]), 2);
    Lambda = abs(diag(positiveEigs));
    % Get a random orthonormal matrix
    [Q, ~] = qr(rand(N));
    % Get a random s.p.d. matrix
    P = Q'*Lambda*Q;
    % =====================================================================
    
    % Perform Cholesky decomposition
    R = chol(P, 'upper');
    
    % Get a random downdate vector
    x = (rand(N, 1) - 0.5).*(10.^((rand(N, 1) - 0.5)));
    
    ticMATLAB = tic;
    RNewMATLAB = cholupdate(R, x, '+');
    timeMATLAB = timeMATLAB + toc(ticMATLAB);
    
    ticMEX = tic;
    RNewMEX = CholeskyUpdateReal(R, x);
    timeMEX = timeMEX + toc(ticMEX);
    
    exactPNew = R'*R + x*x';
    
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
end


fprintf('Total trials:        %12i\n', numTrials);
fprintf('\n');
fprintf('Mean MEX runtime:    %12.4f ms\n', timeMEX*1e3/numTrials);
fprintf('Mean MATLAB runtime: %12.4f ms\n', timeMATLAB*1e3/numTrials);
fprintf('Speedup:             %12.4f x\n', timeMATLAB/timeMEX);
fprintf('\n');
fprintf('Max MEX absolute error:  %.4e\n', max(absErrorMEX));
fprintf('Max MEX relative error:  %.4e\n', max(relErrorMEX));
fprintf('\n');
fprintf('Max MATLAB absolute error:  %.4e\n', max(absErrorMATLAB));
fprintf('Max MATLAB relative error:  %.4e\n', max(relErrorMATLAB));
