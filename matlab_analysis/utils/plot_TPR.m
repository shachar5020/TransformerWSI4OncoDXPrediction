%{
    label_RS - a vector of nx1 with true RS values
    AI_score - a vector of nx1 with predicted RS values
    th - a desired AI_score threshold to display
    
    For each binarizing threshold of AI score, computes and displays TPR 
    and portion of impacted patients
%}

function plot_TPR(label_RS, AI_score, th)

%% compute TPR and impacted patinets at th

TPR_at_th = perfcurve(label_RS, AI_score, 1, ...
                         'XCrit','tpr','TVals', th, 'NBoot',1000);

fprintf('TPR (th=%d) = %.3f (%.3f–%.3f)\n', th,TPR_at_th);

impacted_patients_at_th = sum(AI_score<th)/length(AI_score);
fprintf('%%Impacted Patients (th=%d) = %.1f\n', th, impacted_patients_at_th*100);

%% compute TNR and TPR versus AI score
useBoot = true;
TPR_Y_vec = Compute_TPR_vs_AI(label_RS, AI_score, useBoot);
TPR_X_vec = sort(AI_score);

%% compute %patients versus AI score
n_bins = 500;
edges = linspace(0,100,n_bins+1);
edges_mid = (edges(2:end) + edges(1:end-1))/2;
P = histcounts(AI_score, edges)';

patients_VS_AI = cumsum(P);
patients_VS_AI = patients_VS_AI/max(patients_VS_AI);

Patients_X_vec = edges_mid;
%% display
display_TPR(TPR_X_vec, TPR_Y_vec, Patients_X_vec, patients_VS_AI, th, TPR_at_th(1), impacted_patients_at_th);