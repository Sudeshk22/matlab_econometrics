% Homework3 || Question-5 
%% a)
data= xlsread("C:\Users\sudesh yadav\OneDrive - IIT Kanpur\Documents\MATLAB\Advertising.xlsx")
x = data(:, 1);
y = data(:, 2);
model = fitlm(x, y);
beta = model.Coefficients.Estimate;
std_error_beta = model.Coefficients.SE;
residuals = model.Residuals.Raw;
disp(['Beta: ' num2str(beta(2))]);
disp(['Standard Error of Beta: ' num2str(std_error_beta(2))]);
disp('Residuals:');
disp(residuals);
%%
%% b) 
time = 1:numel(residuals);
plot(time, residuals, 'o');
xlabel('Time');
ylabel('Residuals');
title('Residuals vs. Time');
corr_matrix = corrcoef(time, residuals);
correlation_coefficient = corr_matrix(1, 2);
disp(['Correlation Coefficient: ' num2str(correlation_coefficient)]);

%%
%c)% Assuming you already have residuals from previous steps
	test_statistic = sum(diff(residuals).^2) / sum(residuals.^2);
	
	% Get the number of observations and regressors (including the intercept)
	n = length(residuals);
	k = 2;
	
	% Calculate the degrees of freedom for the test
	df = n - k;
	
	% Compute the critical values for alpha = 0.05
	lower_critical_value = 1.535;  % For positive autocorrelation
	upper_critical_value = 2.465;  % For negative autocorrelation
	
	% Calculate the p-value for the test
	p_value = 2 * (1 - normcdf(abs(test_statistic), 0, 2));
	
	% Display the test statistic and p-value
	disp("Durbin-Watson Test Statistic:");
	disp(test_statistic);
	disp("P-Value:");
	disp(p_value);
	
	% Make the decision and conclusion
	alpha = 0.05;
	if p_value < alpha
	    disp("Conclusion: There is evidence of positive autocorrelation.");
	else
	    disp("Conclusion: There is no significant evidence of positive autocorrelation.");
	end



%% Question - 6 (a)
data= xlsread("C:\Users\sudesh yadav\OneDrive - IIT Kanpur\Documents\MATLAB\TransportChoiceDataset.xlsx")
% Extract variables from the data
	depend = data(:, 6); % Dependent variable (1 if automobile is chosen, 0 if transit is chosen)
	dcost = data(:, 1); % Transit fare minus automobile travel cost ($)
	cars = data(:, 2); % Number of cars owned by the traveler's household
	dovtt = data(:, 3); % Transit out-of-vehicle travel time minus automobile out-of-vehicle travel time (minutes)
	divtt = data(:, 4); % Transit in-vehicle travel time minus automobile in-vehicle travel time (minutes)
	
	% Create a table for regression
	tbl = table(depend, dcost, cars, dovtt, divtt, 'VariableNames', {'depend', 'dcost', 'cars', 'dovtt', 'divtt'});
	
	% Probit regression
	probitModel = fitglm(tbl, 'depend ~ dcost + cars + dovtt + divtt', 'Distribution', 'binomial', 'Link', 'probit');
	
	% Extract coefficients and standard errors
	coefficients = table2array(probitModel.Coefficients(:, 1:2));
	coefficients(:, 2) = coefficients(:, 2) * sqrt(probitModel.Dispersion);
	
	% Create a table for coefficients and standard errors
	variableNames = {'Intercept', 'dcost', 'cars', 'dovtt', 'divtt'};
	tableCoefficients = table(coefficients(:, 1), coefficients(:, 2), 'VariableNames', {'Coefficients', 'Standard_Errors'}, 'RowNames', variableNames);
	
	% Display the table
	disp(tableCoefficients);
%%
    % For Logit Model
    	filename = 'C:\Users\sudesh yadav\OneDrive - IIT Kanpur\Documents\MATLAB\TransportChoiceDataset.xlsx';
	sheet = 1; % Assuming the data is on the first sheet
	
	data = xlsread(filename, sheet);
	
	% Extract variables from the data
	depend = data(:,6); % Dependent variable (1 if automobile is chosen, 0 if transit is chosen)
	dcost = data(:, 1); % Transit fare minus automobile travel cost ($)
	cars = data(:, 2); % Number of cars owned by the traveler's household
	dovtt = data(:, 3); % Transit out-of-vehicle travel time minus automobile out-of-vehicle travel time (minutes)
	divtt = data(:, 4); % Transit in-vehicle travel time minus automobile in-vehicle travel time (minutes)
	
	% Create a table for regression
	tbl = table(depend, dcost, cars, dovtt, divtt, 'VariableNames', {'depend', 'dcost', 'cars', 'dovtt', 'divtt'});
	
	% Logit regression
	logitModel = fitglm(tbl, 'depend ~ dcost + cars + dovtt + divtt', 'Distribution', 'binomial', 'Link', 'logit');
	
	% Extract coefficients and standard errors
	coefficients = table2array(logitModel.Coefficients(:, 1:2));
	coefficients(:, 2) = coefficients(:, 2) * sqrt(logitModel.Dispersion);
	
	% Create a table for coefficients and standard errors
	variableNames = {'Intercept', 'dcost', 'cars', 'dovtt', 'divtt'};
	tableCoefficients = table(coefficients(:, 1), coefficients(:, 2), 'VariableNames', {'Coefficients', 'Standard_Errors'}, 'RowNames', variableNames);
	
	% Display the table
	disp(tableCoefficients);

    %% Qsn 6c
    
% Calculate predicted probabilities
X = table2array(tbl(:, 2:end));
predictedProbabilities = normcdf(probitModel.Fitted.LinearPredictor);

% Calculate log-likelihood
logLikelihood = sum(tbl.depend .* log(predictedProbabilities) + (1 - tbl.depend) .* log(1 - predictedProbabilities));

% Get the number of parameters in the model
numParameters = numel(probitModel.CoefficientNames) - 1;

% Calculate AIC and BIC
AIC = -2 * logLikelihood + 2 * numParameters;
BIC = -2 * logLikelihood + numParameters * log(size(tbl, 1));

% Calculate Hit-rate
predictedLabels = double(predictedProbabilities >= 0.5);
hitRate = sum(predictedLabels == tbl.depend) / numel(tbl.depend) * 100;

% Display the results
fprintf('Probit Model Results:\n');
fprintf('======================\n');
fprintf('Sum of Log-Likelihood: %.3f\n', logLikelihood);
fprintf('AIC: %.3f\n', AIC);
fprintf('BIC: %.3f\n', BIC);
fprintf('Hit-rate: %.2f%%\n', hitRate);

% For Logit Model 
% Calculate predicted probabilities
X = table2array(tbl(:, 2:end));
predictedProbabilities = predict(logitModel, tbl);

% Calculate log-likelihood
logLikelihood = sum(tbl.depend .* log(predictedProbabilities) + (1 - tbl.depend) .* log(1 - predictedProbabilities));

% Get the number of parameters in the model
numParameters = numel(logitModel.CoefficientNames) - 1;

% Calculate AIC and BIC
AIC = -2 * logLikelihood + 2 * numParameters;
BIC = -2 * logLikelihood + numParameters * log(size(tbl, 1));

% Calculate Hit-rate
predictedLabels = double(predictedProbabilities >= 0.5);
hitRate = sum(predictedLabels == tbl.depend) / numel(tbl.depend) * 100;

% Display the results
fprintf('Logit Model Results:\n');
fprintf('=====================\n');
fprintf('Sum of Log-Likelihood: %.3f\n', logLikelihood);
fprintf('AIC: %.3f\n', AIC);
fprintf('BIC: %.3f\n', BIC);
fprintf('Hit-rate: %.2f%%\n', hitRate);
%%