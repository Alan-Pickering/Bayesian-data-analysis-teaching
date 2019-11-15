%bayes_table.m
%v1; 14.11.19
%
%This programme tries to deal with the classic problem originally posed by
%Bayes in the 1760s. He considered throwing samples of billiard balls on a billiard
%table as a means to estimate the position (along the long dimension of the table)
%of a ball previously placed on the table but not now visible (the target ball). 
%Someone who knows the position of the target ball simply reports how many balls,
%in each sample of balls thrown, are further up the table than the
%target ball, and how many fall short.
%
%The programme does Bayesian estimation, with updating over multiple
%trials. The parameter being estimated is theta, the position of the ball
%along the table (the true position is given below by the variable trueposition).
%
%
%to do
%*****
%a) protect against out of range values for theta
%b) put isinfinite protection on various computations
%c) compute Bayes factor for two hypotheses
%e) rewrite to avoid for loops
%f) use a conjugate prior?

clear variables;
clc;

%rng('default');
rng(2000,'twister');

%integration setup
n_int_pts=50; %number of integration points, this is arbitrary but the more the better estimation works
minval=0; %the end of the billiard table
maxval=1; %the far end of the billiard table
int_step=(maxval-minval)./n_int_pts; %integration size step

%initialise arrays, with zeros
xvals=zeros(n_int_pts+1,1);
k=xvals;
prior_dist=xvals;
likelihd=zeros(n_int_pts+1, 1);
marglike=zeros(n_int_pts+1, 1);

%select parameters of the estimation problem
trueposition=-1; %the position of the ball along the table
while trueposition<0 || trueposition>1
    trueposition=input('Enter true position of the target ball (0-1) ');
end;
%estimation parameters
nplots=4; %maximum number of plots
numtrials=-1; %number of repeated estimation trials 
%try 4
while (numtrials<1 || numtrials>100 || (numtrials>nplots-1 && mod(numtrials,nplots)~=0))
    disp('Next, select the number of estimation trials');
    disp(['If > ' num2str(nplots-1) ' estimation trials are requested, then must be divisible by ' num2str(nplots)]);
    numtrials=input('Choose a number between 1 and 100, inclusive '); 
end;
%next chose when to plot out results i.e. every ploit trials
if numtrials<nplots
    plotit=1;
else    
    plotit=numtrials./nplots; %set >numtrials to stop plotting trial by trial
end;
nballs=-1;%number of balls used to do the estimation
%try 20
while nballs<1 || nballs>100
    nballs=input('Enter number of estimation balls to throw per trial (1-100) ');
end;

%setting up prior distributions
%******************************
%we have to set the prior distribution of the parameters, before data is observed
%p(theta), where x is the integration gridding variable (here the
%position on the table between minval and maxval)
priortype=-1; %type of prior distribution of beliefs about ball position
disp(' ');
disp('Enter your choice of prior, plus <Enter>');
disp('1=point prior at 0.5')
disp('2=uniform random distribution between 0 and 1'); 
disp('3=normal distribution with mean=0.5 and s.d.=0.15');
while priortype<1 || priortype>3
    priortype=input('Select value plus <Enter> ');
end;
pointval=0.5;
normmn=0.5;
normsd=0.15;

xctr=0;
for x=minval:int_step:maxval  %#ok<BDSCI>
    xctr=xctr+1; %increment array index
    xvals(xctr,1)=x; %record x
    if priortype==1
        if x==pointval
            prior_dist(xctr,1)=1;
        else
            prior_dist(xctr,1)=0;
        end;
    elseif priortype==2
        prior_dist(xctr,1)=unifpdf(x,minval,maxval); %prior dist values
    elseif priortype==3
        prior_dist(xctr,1)=normpdf(x,normmn,normsd);
    end;    
end;
%normalise the prior 
norm_prior_dist=prior_dist./sum(prior_dist);

est_position=zeros(numtrials,1);
for tr=1:numtrials

    %where does the ball appear to be
    %on the basis that samp% of randomly thrown balls are beyond the true position
    %and 100-samp% are short of the position
    mysample=rand(nballs,1); %uniformly randomly thrown balls used to estimate true position
    nballs_short=sum(mysample<=trueposition);
    est_position(tr)=nballs_short./nballs; %would be a good way to estimate the position, assuming a uniform random dist

    %and next compute the likelihood
    %*******************************
    %recall that ths is the prob of the event given the parameter value
    %p(y|theta)
    %the event y is that nballs_short of the nballs thrown landed short of the true
    %position; what is the likelihood of that occurring for each value of
    %theta? The probability of exactly nballs_short (=k) balls falling short of the true
    %position from the nballs thrown (=n), for the trueposition (=p), is given by the probability mass function of a Binomial
    %distribution, f(k,n,p) = Q*p^k*(1-p)^n-k, where Q is n!/[k!*(n-k)!]
    %this can be quickly computed as binopdf(nballs_short, nballs, theta) over all
    %possible values of theta
    thetactr=0;
    for theta=minval:int_step:maxval  %#ok<BDSCI>
        thetactr=thetactr+1; %increment array index
        likelihd(thetactr,1) = binopdf(nballs_short, nballs, theta);
    end;
    
    %compute the marginal likelihood or evidence
    %*******************************************
    %this is p(y | theta)*p(theta)
    %this is done by computing the product of the likelihood and the prior distribution
    %over all values of theta
    %we approximate the integral here
    thetactr=0;
    for theta=minval:int_step:maxval  %#ok<BDSCI>
        
        thetactr=thetactr+1; %increment array index
        if isfinite(prior_dist(thetactr,1))
            marglike(thetactr,1)=prior_dist(thetactr,1).*likelihd(thetactr,1);
        end
        
    end;
    
    %compute the posterior distribution using Bayes rule
    %***************************************************
    %posterior is p(theta | y)*p(theta)./p(y)  where p(y) is the marginal
    %likelihood summed up, i.e. the evidence
    post_dist=marglike./sum(marglike); %this normalises the posterior probability
    
    %now set prior for next trial to be posterior from previous
    %aka Bayesian updating
    %also aka "belief propagation"
    prior_dist=post_dist;
    
    %plot out posterior distribution
    if tr==numtrials || mod(tr, plotit)==0
        figure;
        plot(xvals,norm_prior_dist,'ok',xvals,post_dist,'xb-');
        tmsg={'Parameter values: Numerically Derived',['After ' num2str(tr) ' trials']};
        title(tmsg);
        xlabel('Theta parameter (estimated position of ball along the table)');
        ylabel('Density');
        %now plot true position
        hold on;
        ht=max(post_dist);
        plot([trueposition, trueposition], [0, ht], 'r-');
        %now plot mean outcomes from the ball throwing
        hold on;
        mn_est=mean(est_position(1:tr));
        plot([mn_est, mn_est], [0, ht], 'g-');
        legend('Initial Prior','Posterior','True value','Mean est. data','Location','best')
   end
    
end;




