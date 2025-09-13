clear all
close all
clc

%%
restoredefaultpath
T_per_patient = readtable('metadata/TCGA_metadata.xlsx');
n = height(T_per_patient);
addpath('utils')

%% clinical features
label_ER = T_per_patient.label_ER;
label_PR = T_per_patient.label_PR;
label_Her2 = T_per_patient.label_Her2;
f_HR_pos_Her2_neg = logical(T_per_patient.HR_pos_HER2_neg);
label_node = T_per_patient.label_node;
TumorSize = T_per_patient.TumorSizeMM/10;
Age = T_per_patient.Age;
Grade = T_per_patient.Grade;
ClinicalRisk = getClinicalRisk(Grade, TumorSize, label_node);

RS = T_per_patient.RS_estimated;
label_RS = RS>=26;

stop
%% RankBased measures (AUROC, AUPRC, AUPRC-neg)

% uncomment for AUROC
YCrit = 'tpr';
XCrit = 'fpr';

% uncomment for AUPRC
% YCrit = 'prec'; % precision = PPV
% XCrit = 'reca'; % recall = TPR = Sensitivity

% uncomment for AUPRC for the negative class
% YCrit = 'npv'; 
% XCrit = 'spec'; % Specificity = TNR

%
close all
clc

AI_score_only_HE = T_per_patient.AI_score;
AI_score_combined = T_per_patient.AI_score_combined_calibrated;

f_valid = f_HR_pos_Her2_neg & ~isnan(RS) & ~isnan(AI_score_combined);

fprintf('# Patients in analysis: %d\n', sum(f_valid));
fprintf('\n')

plot_performance_curve(label_RS(f_valid),AI_score_only_HE(f_valid), AI_score_combined(f_valid), XCrit, YCrit);

%% utility measures

AI_score = T_per_patient.AI_score_combined_calibrated;

f_valid = f_HR_pos_Her2_neg & ~isnan(RS) & ~isnan(AI_score) & ~T_per_patient.UsedForCalibration;

fprintf('# Patients in analysis: %d\n', sum(f_valid));

fprintf('\n')
th_low = 16;
th_high = 26;

plot_TPR(label_RS(f_valid), AI_score(f_valid), th_low);
fprintf('\n')

plot_TNR(label_RS(f_valid), AI_score(f_valid), th_high);
fprintf('\n')

plot_NPV(label_RS(f_valid), AI_score(f_valid), th_low);
fprintf('\n')

plot_PPV(label_RS(f_valid), AI_score(f_valid), th_high);
fprintf('\n')

th = 26;
label_AI = AI_score >= th;

BACC = compute_BACC(label_AI(f_valid), label_RS(f_valid));
fprintf('BACC = %.3f\n', BACC)
fprintf('\n')

F1 = compute_F1(label_AI(f_valid), label_RS(f_valid));
fprintf('F1 = %.3f\n', F1)
fprintf('\n')

F1 = compute_F1(~label_AI(f_valid), ~label_RS(f_valid));
fprintf('F1-neg = %.3f\n', F1)
fprintf('\n')

%% RMSE: calibrated VS uncalibrated
clc

AI_score_uncalibrated = T_per_patient.AI_score_combined;
AI_score_calibrated = T_per_patient.AI_score_combined_calibrated;
fprintf('RMSE without calibration: %.3f \n', rmse(AI_score_uncalibrated(f_valid), RS(f_valid)))
fprintf('RMSE with calibration: %.3f \n', rmse(AI_score_calibrated(f_valid), RS(f_valid)))

%% Box Plot
AI_score = T_per_patient.AI_score_combined_calibrated;
f_valid = f_HR_pos_Her2_neg & ~isnan(RS) & ~isnan(AI_score) & ~T_per_patient.UsedForCalibration;

% Inputs
fprintf('Pearson r = %.3f \n', corr(AI_score(f_valid), RS(f_valid), "Type","Pearson"))
fprintf('Spearman r = %.3f \n', corr(AI_score(f_valid), RS(f_valid), "Type","Spearman"))


close all
eps = 0.001;
bin_edges = linspace(0, 40, 9);
bin_edges(end) = 100;
bin_edges(2:end) = bin_edges(2:end) + eps;
[~, ~, bin_indices] = histcounts(RS(f_valid), bin_edges);

figure;
boxplot(AI_score(f_valid), bin_indices, 'Widths', 0.75, 'Colors','k');
ylim([3 60])
yticks(0:5:60)
grid on;
ylabel('AI score')
xlabel('Oncotype DX RS')
xticklabels({'0-5','5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35+'})
set(gca, 'fontsize', 15)

%% survival

AI_score = T_per_patient.AI_score_combined_calibrated;
times = T_per_patient.Disease_Free_Months/12;
cens = nan(size(T_per_patient,1), 1);
cens(ismember(T_per_patient.Disease_Free_Status, 'DiseaseFree')) = 1;
cens(ismember(T_per_patient.Disease_Free_Status, 'Recurred/Progressed')) = 0;

f_valid = f_HR_pos_Her2_neg & ~isnan(RS) & ~isnan(AI_score) & ~T_per_patient.UsedForCalibration & ~isnan(times) & ~isnan(cens);
th = 26;

str_endpoint = 'Disease-Free Survival Probability';
display_kaplan(times(f_valid), cens(f_valid), AI_score(f_valid), RS(f_valid), th, str_endpoint)

%% Discordances with Clinical Risk
clc
AI_score = T_per_patient.AI_score_combined_calibrated;
f_valid = f_HR_pos_Her2_neg & ~isnan(ClinicalRisk) & ~T_per_patient.UsedForCalibration & ~isnan(AI_score);


% choose an option:
f_valid = f_valid & (Age <= 50) & (label_node == 0); % option 1
% f_valid = f_valid & (Age > 50) & (label_node == 0); % option 2
% f_valid = f_valid & (Age <= 50) & (label_node == 1); % option 3
% f_valid = f_valid & (Age > 50) & (label_node == 1); % option 4

f_clinNeg = f_valid & (ClinicalRisk == 0);
f_clinPos = f_valid & (ClinicalRisk == 1);
f_AINeg = f_valid & (AI_score < 16);
f_AIPos = f_valid & (AI_score >= 26);
f_AIMed = f_valid & (AI_score >= 16) & (AI_score < 26);

fprintf('Total #patients: %d \n', sum(f_valid));
fprintf('\n')

fprintf('#Patients with low clinical risk: %d (%.1f%%)\n', sum(f_clinNeg), sum(f_clinNeg)/sum(f_valid)*100)
fprintf('#Patients with low clinical risk & AI low: %d (%.1f%%)\n', sum(f_clinNeg & f_AINeg), sum(f_clinNeg & f_AINeg)/sum(f_clinNeg)*100)
fprintf('#Patients with low clinical risk & AI med: %d (%.1f%%)\n', sum(f_clinNeg & f_AIMed), sum(f_clinNeg & f_AIMed)/sum(f_clinNeg)*100)
fprintf('#Patients with low clinical risk & AI high: %d (%.1f%%)\n', sum(f_clinNeg & f_AIPos), sum(f_clinNeg & f_AIPos)/sum(f_clinNeg)*100)
fprintf('\n')


fprintf('#Patients with high clinical risk: %d (%.1f%%)\n', sum(f_clinPos), sum(f_clinPos)/sum(f_valid)*100)
fprintf('#Patients with high clinical risk & AI low: %d (%.1f%%)\n', sum(f_clinPos & f_AINeg), sum(f_clinPos & f_AINeg)/sum(f_clinPos)*100)
fprintf('#Patients with high clinical risk & AI med: %d (%.1f%%)\n', sum(f_clinPos & f_AIMed), sum(f_clinPos & f_AIMed)/sum(f_clinPos)*100)
fprintf('#Patients with high clinical risk & AI high: %d (%.1f%%)\n', sum(f_clinPos & f_AIPos), sum(f_clinPos & f_AIPos)/sum(f_clinPos)*100)

