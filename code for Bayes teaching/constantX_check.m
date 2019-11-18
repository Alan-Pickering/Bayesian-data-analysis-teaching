%constantX_check.m
%
%this script simply checks Bayes' 1764 proof that
%the number of balls X=p, p=0,1,2....,n
%lying to the left of a target ball on the table
%is 1/(n+1) for all p
%
%It uses Matlab's symbolic variable processing features 
%and the integration command int to integrate a specified function with respect
%to a symbolic variable

clc;
clear variables;

syms theta; %start symbolic variable processing for theta
n=input('Enter the number of balls thrown '); %number of balls thrown
%initialise arrays
pvals=zeros(n+1,1);
myresult=zeros(n+1,1);
for p=0:n
    myfunX=theta.^p.*(1-theta).^(n-p); %part of the prob mass function of the binomia distribution
    pvals(p+1)=p;
    the_integral=int(myfunX,theta, 0, 1); %integrate over theta between limits of 0 and 1
    myresult(p+1)=eval(the_integral); %convert to a double precison value from a fraction, and store in array
    %use scaling constant based on n!/p!(n-p)! to get right integral
    myresult(p+1)=nchoosek(n,p).*myresult(p+1);
end;
figure;
plot(pvals,myresult,'o-k');
title({['Bayesian Billiard Table with ' num2str(n) ' balls'], ['Theoretical result = ' num2str(1/(n+1))], ' for all values of p'});
axis([0 n 0 1]); %control the plotting axes
xticks(0:1:n); %control tick placement on x-axis
ylabel({'Probability of observing X=p balls' 'to left of target ball'});
xlabel('Number of balls to left of target (=p)');