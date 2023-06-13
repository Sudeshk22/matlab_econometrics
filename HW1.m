x = [4; 1; 2; 3; 3; 4];
y = [16; 5; 10; 15; 13; 22];
% Create the X matrix
X = [ones(6,1), x];
% a) calculating y'y 
yTy = y' * y;
disp("yTy:");
disp(yTy)
% calculating X'X
XTX = X' * X;
disp("XTX:");
disp(XTX)
% calculating the X'y
XTy = X' *y
% displaying the results 
disp("yTy:");

disp(yTy);


% B)
beta_hat = inv(X'*X)* X'*y

% c) 
residual_vector = y - X*beta_hat

%d) 
y_mean = mean(y)
y_hat = X*beta_hat
SSr = sum((y_hat - y_mean).^2)

%e) 
SSE = sum((y_hat - y).^2)

%f)
n = size(X,1)
k = size(X,2)
sigma_hat = SSE/(n-k)
inv(X'*X) *sigma_hat

%g) 
sqrt(diag(inv(X'*X) *sigma_hat))

%h)
% projection matrix 
p = X * inv(X' *X) * X'

% residual generator matrix 
M = eye(6) - p

%i)
% standard error of regresSsion sigma_hat
sqrt(sigma_hat)
% coefficient of determination
% R_square = 1- SSE/TSS
TSS = sum((y - y_mean).^2)
R_sqaure = 1- (SSE)/(TSS)

%j )
y_h= beta_hat(1,1) + 4* beta_hat(2,1)

%-----Question 5-----------
[data,headers] = xlsread("C:\Users\sudesh yadav\OneDrive - IIT Kanpur\Documents\MATLAB\time.xlsx")


%% ------part(a)-------
% extracting the required columns 
Y1 = data(:,2);
X2 = data(:,5);
X3 = data(:,6);

% constructing the design matrix 
X = [X2, X3];

% performing the regression analysis 
model = fitlm(X,Y1) 
% in the result:Root Mean Squared Error: 17.4
%R-squared: 0.843,  Adjusted R-Squared: 0.824
%  Access the coefficient estimates
coefficients = model.Coefficients

% Step 4: Access the standard error of regression
standard_error = model.Residuals

%--------------
%covariance matrix of the estimated beta's 
covMatrix = model.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se = sqrt(diag(covMatrix));
disp(se)

% Calculate the standard error of regression
SE_regression = sqrt(model.MSE);
disp(SE_regression);


%% ---------part(b)------
Y3 = data(:,4)
% performing the regression analysis 
model2 = fitlm(X,Y3) 

% coefficient estimates
coefficients = model2.Coefficients

% standard error of regression
standard_error = model2.Residuals
%covariance matrix of the estimated beta's 
covMatrix2 = model2.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se2 = sqrt(diag(covMatrix2));
disp(se2)

% Calculate the standard error of regression
SE_regression = sqrt(model2.MSE);
disp(SE_regression);


%% -----partc----
year = data(:,1)

%creating the design matrix 
Ct = 1 - (year >= 1939 & year <= 1945);
Dt = (year >= 1939 & year <= 1945);
designMatrix = [Ct, Dt, X2, X3];

% applying the linear model 
model3 = fitlm(designMatrix, Y1,'Intercept', false)

% coefficient estimates
coefficients = model3.Coefficients

% standard error of regression
standard_error = model3.Residuals
%covariance matrix of the estimated beta's 
covMatrix3 = model3.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se3 = sqrt(diag(covMatrix3));
disp(se3)

% Calculate the standard error of regression
SE_regression = sqrt(model3.MSE);
disp(SE_regression);

%% -- part(d)
var = covMatrix3(2,2) + covMatrix3(1,1) - 2*covMatrix3(1,2)

%% -- part(e)
% creating the design matrix 
X = [Dt,X2,X3];
model4 = fitlm(X,Y1)

% coefficient estimates
coefficients = model4.Coefficients

% standard error of regression
standard_error = model4.Residuals
%covariance matrix of the estimated beta's 
covMatrix4 = model4.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se4 = sqrt(diag(covMatrix4));
disp(se4)

% Calculate the standard error of regression
SE_regression = sqrt(model4.MSE);
disp(SE_regression);

%% ----part(f)----
% Create the variables
X4 = X3.*Dt;
X5 = X3 .* Ct;

% creating the design matrix 
X = [Dt,X2,X3,X4];
Y2 = data(:,3)

model5 = fitlm(X,Y2)
% coefficient estimates
coefficients = model5.Coefficients

% standard error of regression
standard_error = model5.Residuals
%covariance matrix of the estimated beta's 
covMatrix5 = model5.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se5 = sqrt(diag(covMatrix5));
disp(se5)

% Calculate the standard error of regression
SE_regression = sqrt(model5.MSE);
disp(SE_regression);

%% ----part(g)----
% creating the design matrix 
X = [Ct,Dt,X2,X4,X5];
model6 = fitlm(X,Y2,'Intercept',false)
% coefficient estimates
coefficients6 = model6.Coefficients

% standard error of regression
standard_error = model6.Residuals
%covariance matrix of the estimated beta's 
covMatrix6 = model6.CoefficientCovariance


% finding the standad errors of the estimates of beta's 
se6 = sqrt(diag(covMatrix6));
disp(se6)

% Calculate the standard error of regression
SE_regression = sqrt(model6.MSE);
disp(SE_regression);

% from the model the absolut 
% % for beta4 case ::
% p value is smaller than 0.05 (significant level) so we 
% we will reject the null hypothesis

% for beta 5 case 
% tstats value pvalue is greater than 0.05 (significant level) so we will 
% fail to reject the null
% hypothesis
%% --- part (h)
X6 = X2 .* Dt;
% creating the design matrix 
X = [Dt,X2,X3,X4,X6];
Y3 = data(:,4);
model7 = fitlm(X, Y3)
% coefficient estimates
coefficients = model7.Coefficients

% standard error of regression
standard_error = model7.Residuals
%covariance matrix of the estimated beta's 
covMatrix7 = model7.CoefficientCovariance

% finding the standad errors of the estimates of beta's 
se7 = sqrt(diag(covMatrix7));
disp(se7)

% Calculate the standard error of regression
SE_regression = sqrt(model7.MSE);
disp(SE_regression);
