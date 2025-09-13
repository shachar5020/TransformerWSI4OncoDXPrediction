%{
    label_RS - a vector of nx1 with true RS values
    AI_score - a vector of nx1 with predicted RS values
    th - a desired AI_score threshold to display
    
    For each binarizing threshold of AI score, computes and displays TNR 
    and portion of impacted patients
%}


function plot_TNR(label_RS, AI_score, th)

%% compute TNR and impacted patinets at th

TNR_at_th = perfcurve(label_RS, AI_score, 1, ...
                         'XCrit','tnr','TVals', th, 'NBoot',1000);

fprintf('TNR (th=%d) = %.3f (%.3f–%.3f)\n', th,TNR_at_th);

impacted_patients_at_th = sum(AI_score>=th)/length(AI_score);
fprintf('%%Impacted Patients (th=%d) = %.1f\n', th, impacted_patients_at_th*100);

%% compute TNR and TPR versus AI score
useBoot = true;
TNR_Y_vec = Compute_TNR_vs_AI(label_RS, AI_score, useBoot);
TNR_X_vec = sort(AI_score);

%% compute %patients
n_bins = 500;
edges = linspace(0,100,n_bins+1);
edges_mid = (edges(2:end) + edges(1:end-1))/2;
P = histcounts(AI_score, edges)';

patients_VS_AI = cumsum(P, 'reverse');
patients_VS_AI = patients_VS_AI/max(patients_VS_AI);

Patients_X_vec = edges_mid;
%% display
display_TNR(TNR_X_vec, TNR_Y_vec, Patients_X_vec, patients_VS_AI, th, TNR_at_th(1), impacted_patients_at_th)