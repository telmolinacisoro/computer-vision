function [M,S]=rigidfactorization_ortho(Zc,epsilon,n_iter)
% Rigid Factorization. The steps are:
% 1. Factorize Z_c using any factorization
% 2. project R into the manifold of motion matrices
% 3. recalculate shape with S^k = {M^k}^\daga Z_c
% 4. recalculate R as Z_c {S_k}^\daga
% 5. if ||M_k - M_{k-1} || < \epsilon finish
% 6. M=M_k S=S_k
% INPUT:
% Zc: 2D tracking data from a monocular video [2F x P], where F is the
% number of frames "n_frames" and P is the number of points "n_points"
% OUTPUT:
% M [2F x 3] camera rotation matrix per frame f
% S [3 x P] rigid shape
% 

if nargin < 2
    epsilon=0.05;
end

% Removing translation component as the mean values of Zc
T = mean(Zc,2);
Zc = Zc - repmat(T,1,size(Zc,2)); 

% Factorize Zc using SVD
[U,D,V] = svd(Zc,'econ');
R = U(:,1:3)*sqrt(D(1:3,1:3));
S = sqrt(D(1:3,1:3))*V(:,1:3)';

%--------------------------------------------------------------------------
% Metric Upgrade Step
F=size(Zc,1);
k=0;
thenorm=epsilon+1;
while thenorm>epsilon && k<n_iter
    Rp=R;
    M=[];
    for f=1:2:F
       Rf=R(f:f+1,:);
       [U2,useless,V2]=svd(Rf,'econ');
       T=U2*V2';
       M=[M;T(1:2,:)];
   end 
    S=pinv(M)*Zc;
    R=Zc*pinv(S);
    k=k+1;
    thenorm=norm(R-Rp,'fro')/numel(M);
end
