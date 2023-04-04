clear
Prior= [110,90;3.33, 0]; %修改1，待求样本的先验信息，即均值和标准差的范围
XM1 = [1,2,3,4];
XM2 = [690,730,760,770,770,780];%修改2，待求样本公式中的的实测参数值
Model_error_mean1=0;Model_error_std_up1 =24;Model_error_std_down1 =0;Model_error_std_interval1=0.25; a1=4.167;b1=-18.310;
Model_error_mean2=0;Model_error_std_up2 =96;Model_error_std_down2 =0;Model_error_std_interval2=1; a2=2.933;b2=-7.218;%修改3待求样本和参数的模型参数
XTM1=log(XM1); N_XM1=length(XTM1);%修改4log化，标贯log化，以及个数
XTM2=log(XM2); N_XM2=length(XTM2);
Upper_Bound_Mu=Prior(1,1); Lower_Bound_Mu=Prior(1,2); Interval_Mu=0.1;%修改5均值上限下限，间隔
Upper_Bound_Sg=Prior(2,1); Lower_Bound_Sg=Prior(2,2); Interval_Sg=0.1;%修改5误差上限下限，间隔
Mu_Array=...
(Lower_Bound_Mu+Interval_Mu/2:Interval_Mu:Upper_Bound_Mu-Interval_Mu/2)';
Sg_Array=...
(Lower_Bound_Sg+Interval_Sg/2:Interval_Sg:Upper_Bound_Sg-Interval_Sg/2)';
Model_error_std_array1=...
(Model_error_std_down1+Model_error_std_interval1/2:Model_error_std_interval1:Model_error_std_up1-Model_error_std_interval1/2)';
Model_error_std_array2=...
(Model_error_std_down2+Model_error_std_interval2/2:Model_error_std_interval2:Model_error_std_up2-Model_error_std_interval2/2)';%均值和标准差等间隔向量
N_intervals_Mu=length(Mu_Array); N_intervals_Sg=length(Sg_Array);N_Model_error_std=length(Model_error_std_array1);%向量个数
ln_Sg=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
ln_Mu=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%ln_Mu生成零向量
Likelihood_Mu1=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
Likelihood_Sg1=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%Likelihood_Mu生成零向量
Likelihood_Mu2=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);
Likelihood_Sg2=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%Likelihood_Mu生成零向量
Inside_Function=zeros(N_intervals_Mu,N_intervals_Sg,N_Model_error_std);%Inside_Function生成零向量
for i=1: N_intervals_Mu
for j=1: N_intervals_Sg 
for k=1: N_Model_error_std
Prior_PDF_Numerical=1/((Prior(1,1)-Prior(1,2))*(Prior(2,1)-Prior(2,2))); 
Model_error_std_PDF1=1/(Model_error_std_up1-Model_error_std_down1);
Model_error_std_PDF2=1/(Model_error_std_up2-Model_error_std_down2);
ln_Sg(i,j,k)=sqrt(log((Sg_Array(j)/Mu_Array(i))^2+1));%从待求样本的先验均值和标准差求ln后的均值
ln_Mu(i,j,k)=log(Mu_Array(i))-0.5*ln_Sg(i,j)^2;%从待求样本的先验均值和标准差求ln后的标准差
Likelihood_Mu1(i,j,k)=a1*ln_Mu(i,j)+b1;%ln（n）的均值和标准差
Likelihood_Sg1(i,j,k)=((a1*ln_Sg(i,j))^2+(Model_error_std_array1(k))^2)^0.5;
Likelihood_Mu2(i,j,k)=a2*ln_Mu(i,j)+b2;%ln（n）的均值和标准差
Likelihood_Sg2(i,j,k)=((a2*ln_Sg(i,j))^2+(Model_error_std_array1(k))^2)^0.5;
Inside_Function(i,j,k)=prod(normpdf(XTM1,Likelihood_Mu1(i,j,k),...%似然函数乘上先验概率
Likelihood_Sg1(i,j,k)))*prod(normpdf(XTM2,Likelihood_Mu2(i,j,k),...%似然函数乘上先验概率
Likelihood_Sg2(i,j,k)))*Prior_PDF_Numerical*Model_error_std_PDF1*Model_error_std_PDF2;
end
end
end

%% Generating Eu samples from Equation (11) using MCMCS 
N_MCMC_XD=30000; MCMC_XD_Original=zeros(N_MCMC_XD,1);%修改6，MCMC样本数
MCMC_XD=zeros(N_MCMC_XD,1);r_XD=zeros(N_MCMC_XD,1);  
Prior_mu_Mean=(Prior(1,1)+Prior(1,2))/2;
Prior_sg_Mean=(Prior(2,1)+Prior(2,2))/2;%先验均值以及标准差均值
Proposal_Cov_XD=Prior_sg_Mean/Prior_mu_Mean;%变异系数
XD0=Prior_mu_Mean;%马尔科夫链初始数据用先验均值
Target_PDF0=sum(sum(sum((lognpdf(XD0,ln_Mu,ln_Sg).*Inside_Function)...%全概率积分（不带常数k），即eu=第一个数时的概率密度
*Interval_Mu*Interval_Sg*Model_error_std_interval1*Model_error_std_interval2)));
MCMC_XD_Original(i,:)=XD0; MCMC_XD(1,:)=XD0;

for i=2:N_MCMC_XD%循环
XD1=random('norm',XD0,XD0*Proposal_Cov_XD);%生成随机数，以先验均值和先验标差
if XD1>0
Target_PDF1=sum(sum(sum((lognpdf(XD1,ln_Mu,ln_Sg).*Inside_Function)...%eu=第二个数时的概率密度
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
junzhi=mean(MCMC_XD)
biaozhuncha=std(MCMC_XD)
aa=1:1:30000
cc=aa.'
DD(:,1)=cc
DD(:,2)=MCMC_XD%用来做散点图
baifenwei5=prctile(MCMC_XD,5)
baifenwei95=prctile(MCMC_XD,95)
%  save sptcpt-vsDATA4.mat
