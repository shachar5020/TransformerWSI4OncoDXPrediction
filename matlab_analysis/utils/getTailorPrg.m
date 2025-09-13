
function [survival_time_years, censored] = getTailorPrg(T_data, prognosis_measure)
n = size(T_data, 1);
if strcmp(prognosis_measure, 'disease-free')
    survival_time_years = T_data.dfs/30.4375/12;
    censored = 1-T_data.dfsind;
elseif strcmp(prognosis_measure, 'distant-recurrence-free')
    survival_time_years = T_data.drfi/30.4375/12;
    censored = 1-T_data.drfiind;
elseif strcmp(prognosis_measure, 'recurrence-free')
    survival_time_years = T_data.rfi/30.4375/12;
    censored = 1-T_data.rfiind;
elseif strcmp(prognosis_measure, 'survival-time_any')
    survival_time_years = T_data.survtime/30.4375/12;
    censored = ones(n, 1);
    f_event = (T_data.survstat == 1); % event when 'dead', no matter the cause
    censored(f_event) = 0;    
elseif strcmp(prognosis_measure, 'survival-time_breast')
    survival_time_years = T_data.survtime/30.4375/12;
    censored = ones(n, 1);
    f_event = (T_data.survstat == 1) & ismember(T_data.cause, {'2', '1'}); % event when 'dead' and when cause of death is 'Breast cancer'.
    censored(f_event) = 0;
else
    err
end