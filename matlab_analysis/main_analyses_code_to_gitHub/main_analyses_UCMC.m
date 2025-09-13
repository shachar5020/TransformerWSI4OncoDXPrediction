clear all
close all
clc

%%
restoredefaultpath
T_per_patient = readtable('metadata/UCMC_metadata.xlsx');
n = height(T_per_patient);
addpath('utils')

%% clinical features
label_ER = T_per_patient.label_ER;
label_PR = T_per_patient.label_PR;
label_Her2 = T_per_patient.label_Her2;
f_HR_pos_Her2_neg = logical(T_per_patient.HR_pos_HER2_neg);
label_node = T_per_patient.label_node;
TumorSize = T_per_patient.TumorSizeMM/10;

StratTumorSize = nan(size(TumorSize));
StratTumorSize(TumorSize > 2) = 2;
StratTumorSize(TumorSize <= 2) = 1;

Grade = T_per_patient.Grade;
Age = T_per_patient.Age;
ClinicalRisk = getClinicalRisk(Grade, TumorSize, label_node);
RecChemo = T_per_patient.RecChemo;
RS = T_per_patient.RS;
label_RS = RS>=26;

cens = 1-T_per_patient.recur;
times = T_per_patient.year_recur;


%% calibrate AI scores
if ~ismember("AI_score_combined_calibrated", T_per_patient.Properties.VariableNames) % run to see how AI scores were calibrated
    f_calibration_samples = T_per_patient.UsedForCalibration;
    AI_score_100_calibration_samples = ...
        T_per_patient.AI_score_combined(f_calibration_samples);
    
    clinical_features = [Grade label_PR label_ER Age StratTumorSize]; 
    clinical_features_100_calibration_samples = clinical_features(f_calibration_samples,:);
        
    Cal_model = load('ClinicoPathology_model_weights.mat');
    
    estimated_RS = clinical_features*Cal_model.beta + Cal_model.a0;
    estimated_RS_100_calibration_samples = estimated_RS(f_calibration_samples);
        
    
    estimated_mean_RS_100_calibration_samples = mean(estimated_RS_100_calibration_samples, 'omitmissing');
    mean_AI_100_calibration_samples = mean(AI_score_100_calibration_samples, 'omitmissing');
    
    scale = estimated_mean_RS_100_calibration_samples/mean_AI_100_calibration_samples;
    
    % calibrate
    AI_score_uncalibrated = T_per_patient.AI_score_combined;
    AI_score_calibrated = AI_score_uncalibrated*scale;
    
    
    T_per_patient.AI_score_combined_calibrated = AI_score_calibrated;
    
    f_valid = f_HR_pos_Her2_neg & ~isnan(RS);
    display_PDD(RS(f_valid), estimated_RS(f_valid));
end
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


%% Utility measures
clc
close all
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

fprintf('Pearson r = %.3f \n', corr(AI_score(f_valid), RS(f_valid), "Type","Pearson"))
fprintf('Spearman r = %.3f \n', corr(AI_score(f_valid), RS(f_valid), "Type","Spearman"))

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

%% survival plots
close all
th = 26;

AI_score = T_per_patient.AI_score_combined_calibrated;

f_valid = f_HR_pos_Her2_neg & ~isnan(RS) & ~isnan(AI_score) & ~T_per_patient.UsedForCalibration & ~isnan(times) & ~isnan(cens);
str_endpoint = 'Recurrence-Free Probability';
display_kaplan(times(f_valid), cens(f_valid), AI_score(f_valid), RS(f_valid), th, str_endpoint)

f_valid_chemo = f_valid & RecChemo == 1;
str_endpoint = 'Recurrence-Free Probability';
display_kaplan(times(f_valid_chemo), cens(f_valid_chemo), AI_score(f_valid_chemo), RS(f_valid_chemo), th, str_endpoint)
title('Chemo')

f_valid_NoChemo = f_valid & RecChemo == 0;
str_endpoint = 'Recurrence-Free Probability';
display_kaplan(times(f_valid_NoChemo), cens(f_valid_NoChemo), AI_score(f_valid_NoChemo), RS(f_valid_NoChemo), th, str_endpoint)
title('No Chemo')

%% Forest plot
AI_score = T_per_patient.AI_score_combined_calibrated;
f_valid = ~isnan(AI_score) & ~isnan(RS) & f_HR_pos_Her2_neg;

subgroups = [
    struct('label','\bf{Overall    }','filter',[],'isHeading',true)
    struct('label','   Overall','filter',true(size(AI_score)), 'isHeading',false)
    
    struct('label','\bf{Menopausal status    }','filter',[],'isHeading',true)
    struct('label','   Premenopausal','filter',(Age>50), 'isHeading',false)
    struct('label','   Postmenopausal','filter',(Age<=50), 'isHeading',false)
    
    struct('label','\bf{Grade    }','filter',[],'isHeading',true)
    struct('label','   Low','filter',(Grade==1), 'isHeading',false)
    struct('label','   Intermediate','filter',(Grade==2), 'isHeading',false)
    struct('label','   High','filter',(Grade==3), 'isHeading',false)
    
    struct('label','\bf{Tumor Size    }','filter',[],'isHeading',true)
    struct('label','   \leq2cm','filter',(TumorSize<=2), 'isHeading',false)
    struct('label','   >2cm','filter',(TumorSize>2), 'isHeading',false)
    
    struct('label','\bf{PR status    }','filter',[],'isHeading',true)
    struct('label','   PR Negative','filter',(label_PR==0), 'isHeading',false)
    struct('label','   PR Positive','filter',(label_PR==1), 'isHeading',false)

    struct('label','\bf{Nodal status    }','filter',[],'isHeading',true)
    struct('label','   Negative','filter',(label_node==0), 'isHeading',false)
    struct('label','   Positive','filter',(label_node==1), 'isHeading',false)

    struct('label','\bf{Adjuvant therapy    }','filter',[],'isHeading',true)
    struct('label','   ET alone','filter',(RecChemo==0), 'isHeading',false)
    struct('label','   ET+Chemo','filter',(RecChemo==1), 'isHeading',false)
];

forest_plot_HR( ...
    AI_score, ...
    RS, ...
    times, ...
    cens, ...
    subgroups, ...
    f_valid);

%% Discordances with Clinical Risk (sanky plots)
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

