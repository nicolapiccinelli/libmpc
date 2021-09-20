close all;
clear all;

num_states = 2;
num_output = 2;
num_inputs = 1;
pred_hor = 10;
ctrl_hor = 5;
ineq_c = pred_hor + 1;
eq_c = 0;

ts = 0.1;

useHardConst = true;

nlobj = nlmpc(num_states, num_output, num_inputs);
nlobj.Ts = ts;
nlobj.PredictionHorizon = pred_hor;
nlobj.ControlHorizon = ctrl_hor;

nlobj.Model.StateFcn = ...
    @(x,u) stateEq(x,u);
nlobj.Model.OutputFcn = ...
    @(x,u) outEq(x,u);
nlobj.Optimization.CustomCostFcn = ...
    @(X,U,e,data) objEq(X,U,e,data);
nlobj.Optimization.CustomIneqConFcn = ...
    @(X,U,e,data) conEq(X,U,e,data);

x0 = [0; 1];
u0 = 0;
y0 = [0; 0];

validateFcns(nlobj, x0, u0);

for i = 1:1e5
    [mv,opt,info] = nlmpcmove(nlobj,x0,u0); 
    
    x0 = stateEq(x0, mv(1));
    
    fprintf('%f\n', x0);
    
    if abs(x0(1)) <= 1e-2 && abs(x0(2)) <= 1e-1
        break;
    end
end
%     mpc::cvec<num_states> modelX, modeldX;
%     modelX[0] = 0;
%     modelX[1] = 1.0;
% 
%     mpc::Common<
%         num_states,
%         num_inputs,
%         num_output,
%         pred_hor,
%         ctrl_hor,
%         ineq_c,
%         eq_c>::Result r;
%     r.cmd[0] = 0.0;
% 
%     for (;;) {
%         r = optsolver.step(modelX, r.cmd);
%         stateEq(modeldX, modelX, r.cmd);
%         modelX += modeldX * ts;
%         if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1) {
%             break;
%         }
%     }

function z = stateEq(x,u)
    z = zeros(2,1);
    z(1) = ((1.0 - (x(2) * x(2))) * x(1)) - x(2) + u(1);
    z(2) = x(1);
end    

function y = outEq(x,u)
    y = zeros(2,1);
    y(1) = x(1);
    y(2) = x(2);
end

function J = objEq(X,U,e,data)
    J = sum(sum(X.^2)) + sum(sum(U.^2));
end

function cineq = conEq(X,U,e,data)
    cineq = U - 0.5;
end