close all;
clear all;

% Trivial unstable discrete LTI SISO model
A = 1.5;
B = 1;
C = 1;
D = 0;

Ts = 0.1;
x0 = 10;
u0 = 0;
y0 = 0;

% NLMPC Parameters
Tph = 10;
Tch = 5;
maxIterations = 10000;

% Objective (cost) function
objEq = @(X,U,e,data) sum(sum(X.^2)) + sum(sum(U.^2));

% Inequality (<=) constraint function
% conIneq = @(X,U,e,data) [U - 0.5; -U - 7];
conIneq = @(X,U,e,data) 0;

% Equality (==) constraint function
conEq = @(X,U,data) 0;

Tnx = size(A,2);
Tnu = size(B,2);
Tny = size(C,1);




nlobj = nlmpc(Tnx, Tny, Tnu);
nlobj.Ts = Ts;
nlobj.PredictionHorizon = Tph;
nlobj.ControlHorizon = Tch;

stateEq = @(x,u) A*x + B*u;
outEq = @(x,u) C*x + D*u;

nlobj.Model.StateFcn = @(x,u) stateEq(x,u);
nlobj.Model.OutputFcn = @(x,u) outEq(x,u);
nlobj.Optimization.CustomCostFcn = @(X,U,e,data) objEq(X,U,e,data);
nlobj.Optimization.CustomIneqConFcn = @(X,U,e,data) conIneq(X,U,e,data);
nlobj.Optimization.CustomEqConFcn = @(X,U,data) conEq(X,U,data);

validateFcns(nlobj, x0, u0);

x = x0;
u = u0;
y = y0;

X = zeros(maxIterations, Tnx);
U = zeros(maxIterations, Tnu);
Y = zeros(maxIterations, Tny);

tic
for i = 1:maxIterations
    Y(i,:) = y';
    U(i,:) = u';
    X(i,:) = x';

    [mv,opt,info] = nlmpcmove(nlobj,x,u); 

    u = mv(1);
    y = outEq(x, u);
    x = stateEq(x, u);

%     fprintf('%1.4f %1.4f\n', x, u);

    if abs(x) <= 1e-4
        break;
    end
end
toc

Y = Y(1:i,:);
U = U(1:i,:);
X = X(1:i,:);

t = (1:size(X, 1)) * Ts;

figure();

subplot(3,1,1);
plot(t, X);
title('State evolution');
xlabel('Time (s)');
ylabel('State');

subplot(3,1,2);
plot(t, U);
title('Input evolution');
xlabel('Time (s)');
ylabel('Input');

subplot(3,1,3);
plot(t, Y);
title('Output evolution');
xlabel('Time (s)');
ylabel('Output');
