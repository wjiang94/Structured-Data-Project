function model = okl(K,Y,Lambda)

%%   OKL
%
%   Output Kernel Learning
%   -----------------------------------------------------------------------
%   Copyright © 2011 Francesco Dinuzzo
% 
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Please report comments, suggestions and bugs to:
%   francesco.dinuzzo@gmail.com
%
%   -----------------------------------------------------------------------
%   Reference:
%
%   F. Dinuzzo, C. S. Ong, P. Gehler, and G. Pillonetto. 
%   Learning Output Kernels with Block Coordinate Descent. 
%   Proceedings of the International Conference on Machine Learning, 2011.
%
%   -----------------------------------------------------------------------
%   Inputs:
%   K: Input kernel matrix: (ell x ell) matrix,
%   Y: Training outputs: (ell x m) matrix.
%   Lambda: Vector of regularization parameters.
%
%   -----------------------------------------------------------------------
%   Outputs:
%   model: Output model. 
%   -----------------------------------------------------------------------
%   Example of usage:
%
%       model = OKL(K,Y,Lambda)
%   

%% Preprocessing
Lambda = sort(Lambda(:),'descend');
sa = size(K);
sy = size(Y);
sl = size(Lambda);
ell = sa(1);
m = sy(2);
Nlambda = sl(1);

%% Constants
MAXIT = 1000;
RELTOL = 0.01;
delta = RELTOL* norm(Y,'fro');

%% Initialize variables
model = struct('L', {}, 'C', {}, 'nit', {}, 'lambda', {}, 'J', {}, 'time', {});
J = zeros(MAXIT,1);
L = eye(m);
C = zeros(ell,m);

%% Perform eigendecomposition of the input kernel matrix
[UX,DX] = eig(K);
dx = max(0,abs(diag(DX))); 
Ytilde = UX'*Y;

%% Main loop
for k=1:Nlambda
    
    tic;
    lambda = Lambda(k);
    fprintf('lambda = %0.5g\n',lambda);
    nit = 0;
    res = norm(Y,'fro');
    
    while(res > delta)
        nit = nit+1;
        
        % Sub-problem w.r.t. C.
        %
        % Solve the Sylvester equation KCL+lambda*C = Y using
        % eigendecomposition of K and L. 
        [UY, DY] = eig(L);
        dy = max(0,abs(diag(DY)));
        Q = Ytilde*UY;
        V = Q./(dx*dy'+lambda);
        C = UX*V*UY';
        
        % Sub-problem w.r.t. L.
        F = V*UY';
        E = DX*F;
        R = E'*E;
        [UE, DE] = eig(R);
        dep = max(0,abs(diag(DE)))+lambda;
        Lp = L;
        P = UE'*(R*L+L'*R'+lambda*E'*F)*UE;
        L = UE*(P./(dep*ones(1,m)+ones(m,1)*dep'))*UE';
        
        % Compute the value of the objective functional
        J(nit) = norm(Y,'fro')^2/(2*lambda) +trace((F/4-Ytilde/(2*lambda))'*E*L);
        
        % Compute the variation of L
        res = norm(L-Lp,'fro');
         
        % Check whether the maximum number of iterations has been reached
        if (nit >= MAXIT)
            disp('Reached maximum number of iterations');
            break;
        end
    end
    
    % Assign outputs
    modelk = struct;
    modelk.L = L;
    modelk.C = C;
    modelk.nit = nit;
    modelk.lambda = lambda;
    modelk.J = J(1:nit);
    modelk.time = toc;
    model(k) = modelk;
end
