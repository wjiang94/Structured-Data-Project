%% Load data
clear 
load data/synthdata

N = 30;
Lambda = logspace(-5,0,N) * sqrt(norm(Y'*K*Y));
MSE  = zeros(N,100);
VMSE  = zeros(N,100);
TIME = zeros(N,100);
m = size(Y,2);

%% Identity baseline
IDVMSE= zeros(N,1);
IDMSE = zeros(N,1);
[UX,DX] = eig(Ktrain);
dx = max(0,abs(diag(DX))); 
for k=1:N
    C = UX*((UX'*Ytrain)./(dx*ones(m,1)'+Lambda(k)));
    IDVMSE(k) = mean(mean((Yvalid-Kvalid*C).^2));
    IDMSE(k) = mean(mean((Y0-Ktest*C).^2));
end
[~, ii] = min(IDVMSE);

%% Output kernel learning with Frobenius norm regularization
model = okl(Ktrain,Ytrain,Lambda);
OKLVMSE= zeros(N,1);
OKLMSE = zeros(N,1);
for k=1:N
    OKLVMSE(k) = mean(mean((Yvalid-Kvalid*model(k).C*model(k).L).^2));
    OKLMSE(k) = mean(mean((Y0-Ktest*model(k).C*model(k).L).^2));
end
[~, jj] = min(OKLVMSE);

%% Plot 
Np = 100;
semilogx(Lambda,OKLMSE,'LineWidth',2);
hold on 
semilogx(Lambda,IDMSE,'r--','LineWidth',2);
xlabel('$\lambda$','Interpreter','latex');
title('MSE');
axis square
legend('OKL','Identity baseline');
YL = ylim;
YL(1) = max(0,0.95*YL(1));
YL(2) = YL(2)*1.05;
ylim(YL);
plot(Lambda(jj)*ones(1,100),linspace(YL(1),YL(2),100),'b--','LineWidth',2);
plot(Lambda(ii)*ones(1,100),linspace(YL(1),YL(2),100),'r--','LineWidth',2);

%% Clear workspace
clear n N i j L C UY k m ell lambda lambdaspace;
