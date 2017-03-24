function model = lrokl(K,Y,Lambda,p)

%%   LROKL
%
%   Low Rank Output Kernel Learning
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
%   F. Dinuzzo, and K. Fukumizu. Learning low-rank output kernels. 
%   JMLR: Workshop and Conference Proceedings.
%   Proceedings of the 3rd Asian Conference on Machine Learning.  
%   20:181–196, 
%   Taoyuan, Taiwan, 2011. 
%
%   -----------------------------------------------------------------------
%   Inputs:
%   K: Input kernel matrix: (ell x ell) matrix,
%   Y: Training outputs: (ell x m) matrix.
%   Lambda: Vector of regularization parameters.
%   p: Rank parameter
%
%   -----------------------------------------------------------------------
%   Outputs:
%   model: Output model. 
%   -----------------------------------------------------------------------
%   Example of usage:
%
%       model = LROKL(K,Y,Lambda,p)
%   
%% Preprocessing
Lambda = sort(Lambda(:),'descend');
sy = size(Y);
sl = size(Lambda(:));
ell = sy(1);
m = sy(2);
Nlambda = sl(1);

%% Constants
MAXIT = 1000;
RELTOL = 0.001;
inc = norm(Y,'fro');
delta = RELTOL*inc;

%% Initialize variables
model = struct('A', {}, 'B', {}, 'nit', {}, 'lambda', {}, 'J', {}, 'time', {});
J = zeros(MAXIT,1);
B = eye(m,p);

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
    inc = norm(Y,'fro');
    
    % Keep the initial B away from zero by re-inizialization
    if (sum(sum(B.^2)) < delta)
        B = eye(m,p);
    end

    while(inc > delta)
        nit = nit+1;
        
        % Sub-problem w.r.t. A.
        %
        % Solve the Sylvester equation KAB'B+lambda*A = YB 
        % via eigendecomposition of K and B'B
        [UY,DY] = eig(B'*B);
        dy = diag(DY);
        Q = Ytilde*B*UY;
        V = Q./(dx*dy'+lambda);  
       
        % Sub-problem w.r.t. B.
        Bp = B;
        E = DX*V;
        P = (Ytilde'*E)/(E'*E+lambda*eye(p));
        B = P*UY';
        
        % Compute the value of the objective functional
        if (gt(ell,m))
            J(nit) = (trace((Ytilde- E*P')'*Ytilde/lambda)+trace(V'*E))/2;
        else
            J(nit) = (trace(Ytilde*(Ytilde- E*P')'/lambda)+trace(V'*E))/2;
        end
        
        % Compute the increment of B
        inc = norm(B-Bp,'fro');
        
        % Check whether the maximum number of iterations has been reached
        if (nit >= MAXIT)
            disp('Reached maximum number of iterations');
            break;
        end
    end
    A = UX*V*UY';
    
    modelk = struct;
    modelk.A = A;
    modelk.B = B;
    modelk.nit = nit;
    modelk.lambda = lambda;
    modelk.J = J(1:nit);
    modelk.time = toc;
    model(k) = modelk;
end
end