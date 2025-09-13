%{
    timeVals - vector of KM plot times (x axis of KM)
    survVals - vector of survival probabilities (y axis of KM)
    N - number of patients
    
    reconstructs the survival times and censoring events from the KM plot

%}

function [T_sim, censor_sim] = event_times_from_survival(timeVals, survVals, N)

% Create the CDF
cdfVals = 1 - survVals;

% Preallocate
T_sim      = zeros(N,1);
censor_sim = zeros(N,1); % 0=event default

% Draw N U(0,1)
U = rand(N,1);

for i = 1:N
    u_i = U(i);
    idxT = find(cdfVals >= u_i, 1, 'first');
    if isempty(idxT)
        T_sim(i) = timeVals(end);
    else
        T_sim(i) = timeVals(idxT);
    end
end

censor_sim(T_sim>=max(timeVals)) = 1;
