
function [] = week2sol_matlab() 
clc

% varargout = csvimport( fileName, varargin )
% house_data_all = csvimport('/Users/venki/Desktop/Coursera/Regression/Week2/kc_house_data.csv');
% train_data = csvimport('/Users/venki/Desktop/Coursera/Regression/Week2/kc_house_train_data.csv');
% test_data = csvimport('/Users/venki/Desktop/Coursera/Regression/Week2/kc_house_test_data.csv');
% %%
% save( 'house_data.mat','house_data_all')
% save('train_data.mat','train_data')
% save('test_data.mat','test_data')

%%

load( 'house_data.mat','house_data_all')
load('train_data.mat','train_data')
load('test_data.mat','test_data')


%%
features = {'sqft_living'};
[X] = get_data_matrix(train_data,features,1);
% get output using same function
output = {'price'};
[y] = get_data_matrix(train_data,output,0);
%%

theta_ini=[-47000; 1];
alpha=7e-12;
[ weights,RSS] = gradient_descent(X, theta_ini, y, alpha,2.5e7);
disp(['Weight using training data ' num2str(weights(2))])
disp(['RSS Model 1 using training data ' num2str(RSS)])
%% x train data
% [X_train] = get_data_matrix(train_data,features,1);
% 
% [predictions] = predict_output(X_train, weights);
% 
% output = {'price'};
% [y_train] = get_data_matrix(train_data,output,0);
% 
% %%
% disp(['predicted value - training data ', num2str(predictions(1))])
% disp(['Actual value - training data ', num2str(y_train(1))])


%%

[X_test] = get_data_matrix(test_data,features,1);

[predictions] = predict_output(X_test, weights);

output = {'price'};
[y_test] = get_data_matrix(test_data,output,0);

%%
disp(['predicted value - test data ', num2str(predictions(1))])
disp(['Actual value - test data ', num2str(y_test(1))])

%%  for model 2




features = {'sqft_living' 'sqft_living15'};
[X] = get_data_matrix(train_data,features,1);
output = {'price'};
[y] = get_data_matrix(train_data,output,0);
%%

theta_ini=[-100000; 1; 1];
alpha=4e-12;
[ weights_mul,RSS] = gradient_descent(X, theta_ini, y, alpha,1e9);

%%

[X_test] = get_data_matrix(test_data,features,1);

[predictions] = predict_output(X_test, weights_mul);

output = {'price'};
[y_test] = get_data_matrix(test_data,output,0);

%%
disp(['predicted value model 2- test data ', num2str(predictions(1))])
disp(['Actual value - test data ', num2str(y_test(1))])
disp(['RSS Model 2 ' num2str(RSS)])



end





function [X] = get_data_matrix(house_data,features,const)

for i = 1:length(features)
   idx= strcmp(features{1,i},house_data(1,:))==1;
    X(:,i) = house_data(2:end,idx);
    
end
X=cell2mat(X);
% also add constant term to the data
if const==1
X = [ones(length(X),1) X];
end
end



function [ weights,RSS ] = gradient_descent(feature_matrix, weights, output, alpha,tol)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


converged=0;
itr=1;

while converged ~=1

predictions = predict_output(feature_matrix, weights);
errors = predictions - output;

    for i=1:length(weights) % loop over features

        derivative(i) = feature_derivative(errors,feature_matrix(:, i));
        weights(i)=weights(i)-(alpha)*derivative(i);
    end
    
    gradient_sum_squares=sum(derivative.^2);
    gradient_magnitude = (sqrt(gradient_sum_squares));
    if ~(mod(itr,1000))
    disp(num2str(gradient_magnitude/tol)) 
    end
    
     
    if gradient_magnitude < tol
       converged =1;
       RSS= sum(errors.^2);
    end
    
    itr=itr+1;
end


end




function [predictions] = predict_output(feature_matrix, weights)
% This function gives predictions

predictions = feature_matrix*weights;

end


function[feature_derivative]= feature_derivative(errors, feature)
% This function do the derivative

feature_derivative=2*sum(errors.*feature);

end

