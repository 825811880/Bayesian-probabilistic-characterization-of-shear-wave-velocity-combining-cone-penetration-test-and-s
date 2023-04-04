clear
Prior= [110,90;3.33, 0]; %
XM1 = [1,2,3,4];
XM2 = [690,730,760,770,770,780];%
Model_error_mean1=0;Model_error_std_up1 =24;Model_error_std_down1 =0;Model_error_std_interval1=0.25; a1=4.167;b1=-18.310;
Model_error_mean2=0;Model_error_std_up2 =96;Model_error_std_down2 =0;Model_error_std_interval2=1; a2=2.933;b2=-7.218;%
XTM1=log(XM1); N_XM1=length(XTM1);%
XTM2=log(XM2); N_XM2=length(XTM2);
Upper_Bound_Mu=Prior(1,1); Lower_Bound_Mu=Prior(1,2); Interval_Mu=0.1;%
Upper_Bound_Sg=Prior(2,1); Lower_Bound_Sg=Prior(2,2); Interval_Sg=0.1;%
Mu_Array=...
(Lower_Bound_Mu+Interval_Mu/2:Interval_Mu:Upper_Bound_Mu-Interval_Mu/2)';
Sg_Array=...
(Lower_Bound_Sg+Interval_Sg/2:Interval_Sg:Upper_Bound_Sg-Interval_Sg/2)';
Model_error_std_array1=...
(Model_error_std_down1+Model_error_std_interval1/2:Model_error_std_interval1:Model_error_std_up1-Model_error_std_interval1/2)';
Model_error_std_array2=...
(Model_error_std_down2+Model_error_std_interval2/2:Model_error_std_interval2:Model_error_std_up2-Model_error_std_interval2/2)';%
N_intervals_Mu=length(Mu_Array); N_intervals_Sg=length(Sg_Array);N_Model_error_std=length(Model_error_std_array1);%
ln_Sg=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
ln_Mu=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%
Likelihood_Mu1=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
Likelihood_Sg1=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%
Likelihood_Mu2=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
Likelihood_Sg2=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%
Inside_Function=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%
for i=1: N_intervals_Mu
for j=1: N_intervals_Sg 
for k=1: N_Model_error_std
Prior_PDF_Numerical=1/((Prior(1,1)-Prior(1,2))*(Prior(2,1)-Prior(2,2))); 
Model_error_std_PDF1=1/(Model_error_std_up1-Model_error_std_down1);
Model_error_std_PDF2=1/(Model_error_std_up2-Model_error_std_down2);
ln_Sg(i,j,k)=sqrt(log((Sg_Array(j)/Mu_Array(i))^2+1));%
ln_Mu(i,j,k)=log(Mu_Array(i))-0.5*ln_Sg(i,j)^2;%
Likelihood_Mu1(i,j,k)=a1*ln_Mu(i,j)+b1;%
Likelihood_Sg1(i,j,k)=((a1*ln_Sg(i,j))^2+(Model_error_std_array1(k))^2)^0.5;
Likelihood_Mu2(i,j,k)=a2*ln_Mu(i,j)+b2;%
Likelihood_Sg2(i,j,k)=((a2*ln_Sg(i,j))^2+(Model_error_std_array1(k))^2)^0.5;
Inside_Function(i,j,k)=prod(normpdf(XTM1,Likelihood_Mu1(i,j,k),...%
Likelihood_Sg1(i,j,k)))*prod(normpdf(XTM2,Likelihood_Mu2(i,j,k),...%
Likelihood_Sg2(i,j,k)))*Prior_PDF_Numerical*Model_error_std_PDF1*Model_error_std_PDF2;
end
end
end

%%
N_MCMC_XD=30000; MCMC_XD_Original=zeros(N_MCMC_XD,1);%
MCMC_XD=zeros(N_MCMC_XD,1);r_XD=zeros(N_MCMC_XD,1);  
Prior_mu_Mean=(Prior(1,1)+Prior(1,2))/2;
Prior_sg_Mean=(Prior(2,1)+Prior(2,2))/2;%
Proposal_Cov_XD=Prior_sg_Mean/Prior_mu_Mean;%
XD0=Prior_mu_Mean;%
Target_PDF0=sum(sum(sum((lognpdf(XD0,ln_Mu,ln_Sg).*Inside_Function)...%
*Interval_Mu*Interval_Sg*Model_error_std_interval1*Model_error_std_interval2)));
MCMC_XD_Original(i,:)=XD0; MCMC_XD(1,:)=XD0;

for i=2:N_MCMC_XD%
XD1=random('norm',XD0,XD0*Proposal_Cov_XD);%
if XD1>0
Target_PDF1=sum(sum(sum((lognpdf(XD1,ln_Mu,ln_Sg).*Inside_Function)...%
*Interval_Mu*Interval_Sg*Model_error_std_interval1*Model_error_std_interval2)));
Proposal_PDF_XD0_XD1=normpdf(XD0,XD1,XD1*Proposal_Cov_XD);
Proposal_PDF_XD1_XD0=normpdf(XD1,XD0,XD0*Proposal_Cov_XD); 
r_XD(i,:)=...
Target_PDF1*Proposal_PDF_XD0_XD1/(Target_PDF0*Proposal_PDF_XD1_XD0);
if r_XD(i,:)>rand()
XD0=XD1;Target_PDF0=Target_PDF1;
else
 XD0=XD0;Target_PDF0=Target_PDF0;
end
MCMC_XD_Original(i,:)=XD1;MCMC_XD(i,:)=XD0; 
else
XD0=XD0;Target_PDF0=Target_PDF0;
MCMC_XD_Original(i,:)=XD1;MCMC_XD(i,:)=XD0;
end
end

