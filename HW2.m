% Econometrics HW2 
% Question 5 
clear all
clc 
cd 
[data,headers]= xlsread("C:\Users\sudesh yadav\OneDrive - IIT Kanpur\Documents\MATLAB\ProdFuncData.xlsx")
Y = data(:,2); % output 
L = data(:,3); % labor
K = data(:,4); % capital

% Taking the logarithms of the variables 
logY = log(Y);
logL = log(L);
logK = log(K);

% Question1 (a)
% design matrix will be 
X = [ones(size(logY)), logL, logK];

% Estimate the Cobb-Douglas production function using multiple linear regression
beta = regress(logY, X);

%calculating the predicted values 
Yhat = exp(X * beta);

% calculate the residuals 
residuals = logY - X * beta;
% Calculate the covariance matrix of beta
covMatrix = (residuals' * residuals) / (size(X, 1) - size(X, 2)) * inv(X' * X);

% Calculate the R-square
SSR = sum(residuals.^2);
SST = sum((logY - mean(logY)).^2);
R2 = 1 - SSR / SST;

% Dispaly the regression results as the table format 
disp('cobb douglas Production function: ')
disp('-----------------------------------------------')
disp('variable  coefficient std.error t-stat')
disp(['Intercept  ' num2str(beta(1)) '   ' num2str(sqrt(covMatrix(1, 1))) '   ' num2str(beta(1) / sqrt(covMatrix(1, 1)))])
disp(['log L      ' num2str(beta(2)) '   ' num2str(sqrt(covMatrix(2, 2))) '   ' num2str(beta(2) / sqrt(covMatrix(2, 2)))])
disp(['log K      ' num2str(beta(3)) '   ' num2str(sqrt(covMatrix(3, 3))) '   ' num2str(beta(3) / sqrt(covMatrix(3, 3)))])
disp('-----------------------------------------------')
disp(['R-square: ' num2str(R2)])

% dispaly the cavariance matrix of the beta 
disp('covariance matrix of the beta matrix')
disp('------------------------------------')
disp(covMatrix)

%%----------------------------------%%
% Question 5 (b)
alpha = 0.05;
hat_beta_L = beta(2);
hat_beta_K = beta(3);

%calculating the variance of beta_L + beta_K
var_beta_L = covMatrix(2,2);
var_beta_K = covMatrix(3,3);
cov_beta_LK = covMatrix(2,3);

var_beta_sum = var_beta_L + var_beta_K + 2*cov_beta_LK;
% Set the null hypothesis value
h0 = 1;

% Calculate the test statistic
standard_error = sqrt(var_beta_sum);
t_stat = (hat_beta_L + hat_beta_K - h0) / (standard_error); 

% Calculate the critical value
df = size(X,1) - size(X,2);  % Replace 'degrees_of_freedom' with the appropriate value
t_critical = tinv(1 - alpha/2, df);
% Perform the test
if abs(t_stat) > t_critical
    disp('Reject the null hypothesis (H0)');
else
    disp('Fail to reject the null hypothesis (H0)');
end

% Display the results
disp(['t-statistic: ', num2str(t_stat)]);
disp(['ta-critical: ', num2str(t_critical)]);


%% Question-5(c)
% Take the logarithm of the variables
logY = log(Y);
logL = log(L);
logK = log(K);

% Construct the design matrix
X = [ones(size(logY)), logL, logK, 0.5*(logL.^2), 0.5*(logK.^2), logL.*logK];

% Estimate the translog model using multilinear regression model
beta = regress(logY, X);

% calculated the predicted values 
Y_hat = exp(X* beta);

% calculate the residuals 
residuals = logY - X*beta;

% calculate the covariance matrix of beta 
covMatrix = (residuals'*residuals) / (size(X, 1) - size(X, 2)) * inv(X' * X);

% Calculate the R-square
SSR = sum(residuals.^2);
SST = sum((logY - mean(logY)).^2);
R2 = 1 - SSR / SST;

% Display the regression results
disp('Translog Model:')
disp('----------------------')
disp('Variable     Coefficient     Std. Error     t-Statistic')
disp('-----------------------------------------------------')
disp(['Intercept    ' num2str(beta(1)) '       ' num2str(sqrt(covMatrix(1, 1))) '      ' num2str(beta(1) / sqrt(covMatrix(1, 1)))])
disp(['log L        ' num2str(beta(2)) '       ' num2str(sqrt(covMatrix(2, 2))) '      ' num2str(beta(2) / sqrt(covMatrix(2, 2)))])
disp(['log K        ' num2str(beta(3)) '       ' num2str(sqrt(covMatrix(3, 3))) '      ' num2str(beta(3) / sqrt(covMatrix(3, 3)))])
disp(['(logL)^2     ' num2str(beta(4)) '       ' num2str(sqrt(covMatrix(4, 4))) '      ' num2str(beta(4) / sqrt(covMatrix(4, 4)))])
disp(['(logK)^2     ' num2str(beta(5)) '       ' num2str(sqrt(covMatrix(5, 5))) '      ' num2str(beta(5) / sqrt(covMatrix(5, 5)))])
disp(['logL*logK    ' num2str(beta(6)) '       ' num2str(sqrt(covMatrix(6, 6))) '      ' num2str(beta(6) / sqrt(covMatrix(6, 6)))])
disp('-----------------------------------------------------')
disp(['R-square: ' num2str(R2)])

% Display the covariance matrix of beta
disp('Covariance Matrix of beta:')
disp('--------------------------')
disp(covMatrix)

%% question 5 (d)


% Take the logarithm of the variables
logY = log(Y);
logL = log(L);
logK = log(K);

% Construct the design matrix for the unrestricted model
X_unrestricted = [ones(size(logY)), logL, logK, 0.5*(logL.^2), 0.5*(logK.^2), logL.*logK];

% Estimate the unrestricted Cobb-Douglas production function using multiple linear regression
beta_unrestricted = regress(logY, X_unrestricted);
Yhat_unrestricted = exp(X_unrestricted * beta_unrestricted);
residuals_unrestricted = logY - X_unrestricted * beta_unrestricted;
SSR_unrestricted = sum(residuals_unrestricted.^2);

% Construct the design matrix for the restricted model
X_restricted = [ones(size(logY)), logL, logK];
beta_restricted = regress(logY, X_restricted);
Yhat_restricted = exp(X_restricted * beta_restricted);
residuals_restricted = logY - X_restricted * beta_restricted;
SSR_restricted = sum(residuals_restricted.^2);

% Calculate the degrees of freedom
n = size(X_unrestricted, 1);
k_unrestricted = size(X_unrestricted, 2) ;
k_restricted = size(X_restricted, 2) ;
df_numerator = k_unrestricted - k_restricted;
df_denominator = n - k_unrestricted;

% Calculate the F-statistic
F_statistic = ((SSR_restricted - SSR_unrestricted) / df_numerator) / (SSR_unrestricted / df_denominator)

% Set the significance level (e.g., alpha = 0.05)
alpha = 0.05;

% Calculate the critical value based on the significance level and degrees of freedom
criticalValue = finv(1 - alpha, df_numerator, df_denominator) 

% Perform the F-test
if F_statistic > criticalValue
    disp('Reject the null hypothesis: The Cobb-Douglas model is not appropriate.')
else
    disp('Fail to reject the null hypothesis: The Cobb-Douglas model is appropriate.')
end

%% question5(e)
%%   Qsn 5 e

X = [ones(size(logY)), logL, logK, 0.5*(logL.^2), 0.5*(logK.^2), logL.*logK];
R = [0 1 1 0 0 0; 0 0 0 1 1 2] ;
q = [1; 0]                     ;


hat_beta = regress(logY,X) ;
Res = logY- X*hat_beta ;

n = size(logY,1);
k = size(X,2)   ;
L =  size(R,1)  ;

SSR = sum(Res.^2) ;
sigma2 = SSR/(n-k) ;

F_stat= ((R*hat_beta-q)'*inv(sigma2*R*inv(X'*X)*R')*(R*hat_beta-q))/L

alpha = 0.05;
df1 = 2;   % Numerator degrees of freedom
df2 = 21;  % Denominator degrees of freedom

criValue = finv(1-alpha, df1, df2);


% Perform the F-test
if F_stat > criValue
    disp('Reject the null hypothesis:')
else
    disp('Fail to reject the null hypothesis: ')
end



