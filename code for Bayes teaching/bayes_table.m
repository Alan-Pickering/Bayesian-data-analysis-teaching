%bayes_table.m
%v2; 16.11.19
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
%it can also do the example, using just a single trial, illustrated in a
%Royal Society video by David Spiegelhalter in conversation with Brian Cox
%
%https://royalsociety.org/about-us/programmes/people-of-science/david-spiegelhalter-bayes-fisher/
%
%You can select to run this single trial example, or take control of the
%simulation parameters.
%
%
%to do in later versions
%***********************
%a) protect against out of range values for theta
%b) put "isinfinite" checks in, as protection on various computations, if required
%c) rewrite to avoid for loops
%d) use a conjugate prior to illustrate way to avoid the integration
%e) compute Bayes factor for two comparing hypotheses about the true position

clear variables;
clc;

%rng('default');
rng(2000,'twister');

%integration setup
n_int_pts=50; %number of integration intervals, this is arbitrary but the more the better estimation works
minval=0; %the end of the billiard table
maxval=1; %the far end of the billiard table
int_step=(maxval-minval)./n_int_pts; %integration size step

%initialise arrays, with zeros
xvals=zeros(n_int_pts+1,1);
prior_dist=xvals;
likelihd=zeros(n_int_pts+1, 1);
marglike=zeros(n_int_pts+1, 1);

nplots=4; %maximum number of plots produced by this script

%chance to opt for the Spiegelhalter single trial example, or not
spiegel=0;
while spiegel~=1 && spiegel~=2
    spiegel=input('Do you want the Spiegelhalter example (1) or set your own parameters (2) ');
end;

%select parameters of the estimation problem
%trueposition is the position of the ball along the table
%taking values in the range [0,1]
if spiegel==1
    trueposition=0.5; %the true position is irrelevant in this example
elseif spiegel==2
    trueposition=-1; %to force a choice
end;
while trueposition<0 || trueposition>1
    trueposition=input('Enter true position of the target ball (0-1) ');
end;

%estimation parameters, numtrials and nballs
%numtrials is the number of estimation trials(sets of balls thrown on the
%table to estimate the position of the target ball)
if spiegel==1
    numtrials=1; %force a single trial for this example
elseif spiegel==2
    numtrials=-1; %to force a choice
end;
while (numtrials<1 || numtrials>100 || (numtrials>nplots-1 && mod(numtrials,nplots)~=0))
    disp('Next, select the number of estimation trials');
    disp(['If > ' num2str(nplots-1) ' estimation trials are requested, then must be divisible by ' num2str(nplots)]);
    numtrials=input('Choose a number between 1 and 100, inclusive '); 
end;
%nballs is the number of balls used on each trial to do the estimation
if spiegel==1
    nballs=5; %force the use of 5 balls for this example
elseif spiegel==2
    nballs=-1; %to force a choice
end;
while nballs<1 || nballs>100
    nballs=input('Enter number of estimation balls to throw per trial (1-100) ');
end;

%next chose when to plot out results i.e. every plotit trials
if numtrials<nplots
    plotit=1; %plot every trial if fewer trials than nplots
else    
    plotit=numtrials./nplots; 
end;

%setting up prior distributions
%******************************
%we have to set the prior distribution of the parameter, before data is observed
%theta is the parameter, the position in range [0,1] along the table
%the prior distribution is a prob density function (PDF) for theta
%priortype controls the type of prior distribution of beliefs about ball position
if spiegel==1
    priortype=2; %to use a uniform random distribution for this example
    maxpriortypechoices=2; %needed to make while statement below run without crashing
elseif spiegel==2
    priortype=-1; %to force a choice
    disp(' ');
    disp('Enter your choice of prior, plus <Enter>');
    disp('1=point prior at 0.5')
    disp('2=uniform random distribution between 0 and 1');
    disp('3=normal distribution with mean=0.5 and s.d.=0.15');
    maxpriortypechoices=3; %change as you add more choices
end;
while priortype<1 || priortype>maxpriortypechoices
    priortype=input('Select value plus <Enter> ');
end;
%values which will be used for various priortypes, if selected
pointval=0.5;
normmn=0.5;
normsd=0.15;
%now we will create an approximation of hte prior density
%using x as the gridding variable which divides up
%position on the table (between minval and maxval)
%into n_int_pts+1 equally spaced points
xctr=0;
for x=minval:int_step:maxval  %#ok<BDSCI>
    xctr=xctr+1; %increment array index
    xvals(xctr,1)=x; %record these integration values for use in later plotting
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
    %on the basis that nballs_short of randomly thrown balls fall short of the true position
    %and 100-nballs_short end up beyond the true position
    %this is determined randomly on each estimation trial and is computed
    %in relation to the real trueposition
    mysample=rand(nballs,1); %uniformly randomly thrown balls used to estimate true position
    nballs_short=sum(mysample<=trueposition);
    est_position(tr)=nballs_short./nballs; %would be a good way to estimate the position, assuming a uniform random dist
    %the next two lines are for a one-trial demonstration a la David Spiegelhalter
    if spiegel==1
        %the values in the video example
        %and override the randomly selected values
        est_position(1)=0.4;
        nballs_short=2;
    end;
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
    %for convenience we use the same integration set u[p as we
    %used earlier with xvals
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
    if tr<numtrials
        prior_dist=post_dist; %do not reset prior on final trial
    end;
    
    %plot out posterior distribution
    if tr==numtrials || mod(tr, plotit)==0
        figure;
        %note that we use the stored xvals for plotting
        %these serve as theta values as well, because we used the same
        %integration setup throughout the programme
        plot(xvals,norm_prior_dist,'ok',xvals,post_dist,'xb-');
        tmsg={'Parameter values: Numerically Derived',['After ' num2str(tr) ' trials']};
        title(tmsg);
        xlabel('Theta parameter (estimated position of ball along the table)');
        ylabel('Density');
        ht=max(post_dist); %to control the height of the vertical lines
        %now plot true position
        %not relevant in the Spiegelhalter example
        if spiegel==2
            hold on;
            plot([trueposition, trueposition], [0, ht], 'r-');
        end;
        %now plot the point estimates from the ball throwing
        hold on;
        %first compute the maximum likelihood estimate
        ML_est=mean(est_position(1:tr));
        plot([ML_est, ML_est], [0, ht], 'g-');
        %and the weighted mean of the posterior distribution gives us
        %the Bayesian point estimate
        BPE=sum(xvals.*post_dist);
        hold on;
        plot([BPE, BPE], [0, ht], 'm-');
        if spiegel==1
            %no true position
            legend('Initial Prior','Posterior','Max like est.','Bayesian est.','Location','best')
        elseif spiegel==2
            %includes true position
            legend('Initial Prior','Posterior','True value','Max like est.','Bayesian est.','Location','best')
        end;
    end;
    
end;

%and now check the value is as described in the video
if spiegel==1
    disp(['The best Bayesian point estimate of the position of the target ball= ' num2str(BPE)]);
    disp('This is marked by the vertical line in magenta in the figure.');
    disp('In the Spiegelhalter video this is described as 3/7. Let''s check:-');
    disp(['3/7 = ' num2str(3/7)]);
end;




