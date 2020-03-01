%function to call regression and output the value 
function [w, b, obj, cvErrs,RMSE,RMSEcv,yhat,l,RMSEtrain, RMSEcvtrain] = solution()
 D = csvread("valData.csv",0,1);
 trLb = csvread("valLabels.csv",0,1);

 X=D';
 y=trLb;
 l=[0.01, 0.1, 1, 10, 100, 1000];
 for k=1:6
   [w, b, obj, cvErrs] = ridgeReg(X, y, l(k)); 
   yhat=(X'*w)+b;
   yerror=y-yhat;
   RMSE(k)=RootMeanSquare(yerror);
   RMSEcv(k)=RootMeanSquare(cvErrs);
   disp(k);
 end
 x=l;
 Dtraining = csvread("trainData.csv",0,1);
 trLbtraining = csvread("trainLabels.csv",0,1);
 X1=Dtraining';
 y1=trLbtraining;
 for m=1:6
   [wtrain, btrain, objtrain, cvErrstrain] = ridgeReg(X1, y1, l(m)); 
   yhattrain=(X1'*wtrain)+btrain;
   yerrortrain=y1-yhattrain;
   RMSEtrain(m)=RootMeanSquare(yerrortrain);
   RMSEcvtrain(m)=RootMeanSquare(cvErrstrain);
   disp(m);
 end
 plot(log(x),RMSE,log(x),RMSEtrain,log(x),RMSEcvtrain); 
 xlabel("log(lambda)");
 ylabel("RMSE");
 legend("Validation RMSE","Training RMSE","LOOCV");
 text(100,3,'\downarrow RMSE');
 text(100,103,'\downarrow RMSEcv');

 
end

%Function for ridge regression
function [w, b, obj, cvErrs] = ridgeReg(X, y, l)
k=size(X,1);
n=size(X,2);
I=eye(k);
OnesVect=ones(1,n);
ZeroVector=zeros(k,1);
Ibarfirst=horzcat(I,ZeroVector);
Ibarlastrow=horzcat(transpose(ZeroVector),0);
Ibar=[Ibarfirst;Ibarlastrow];
Xbar=[X;OnesVect];
XTX=Xbar*Xbar';
lIdentity=l*Ibar;
C=XTX+lIdentity;
d=Xbar*y;
wbar=C\d;
b=wbar(end);
w=wbar(1:k);
invC=inv(C);
for i = 1:n
    Xbari=Xbar(:,i);
    yi=y(i);
    numer=(wbar'*Xbari)-yi;
    denom=1-(Xbari'*invC*Xbari);
    cvErrs(i)=numer/denom; 
end
objsum=0;
obj=0;

%Calulating obj
for j = 1:n
    Xj=X(:,j);
    yj=y(j);
    objpart=(w'*Xj+b-yj)^2;
    objsum=objsum+objpart;
end
obj=l*norm(w.^2)+objsum;
%calculating regression term
% regterm=l*sum(w.^2,'all');
end
function [RMSEval]= RootMeanSquare(Z)
Squarederror=Z.^2;
MeanTotalSquarederror=mean(Squarederror);
RMSEval=sqrt(MeanTotalSquarederror);
end

%function used to predict value. Here X was passed as values from test file
function [ytest]=predict(X,w,b)
yhat=(X'*w)+b;
ytest=yhat;
end
