close all
clear all;
clc

% choose and option
Men = 0; %pre menopausals
% Men = 1; %post menopausals

Clin = 'Any'; % Any clinical risk
% Clin = 'Low'; % only low clinical risk
% Clin = 'High'; % only high clinical risk

%%

T_per_patient = readtable('metadata/TAILORx_metadata.xlsx');

n = size(T_per_patient,1);

restoredefaultpath
addpath('utils/')

%% clinical variables

Arm = T_per_patient.rxarm;
InAnalysis = T_per_patient.InAnalysis;
RS = T_per_patient.RS;
RecChemo = T_per_patient.RecChemo;
meno = ismember(T_per_patient.meno, 'Post');

Grade = nan(n,1);
Grade(ismember(T_per_patient.Grade, 'Low')) = 1;
Grade(ismember(T_per_patient.Grade, 'Med')) = 2;
Grade(ismember(T_per_patient.Grade, 'High')) = 3;
TumorSize = T_per_patient.TumorSize/10;
ClinicalRisk = getClinicalRisk(Grade, TumorSize, zeros(size(Grade)));

AI_score = T_per_patient.AI_score_combined;

%% draw kaplan
% prognosis_measure = 'distant-recurrence-free';
% prognosis_measure = 'recurrence-free';
prognosis_measure = 'disease-free';

[times, cens] = getTailorPrg(T_per_patient, prognosis_measure);
f_valid_1 = ~isnan(times) & ~isnan(cens) & ~isnan(AI_score);
f_valid_1 = f_valid_1 & (InAnalysis == true);


if Men == 0
    f_valid = f_valid_1 & (16<= RS) & (26 > RS);
    f_valid = f_valid & (meno==0);
elseif Men == 1
    f_valid = f_valid_1 & (11<= RS) & (26 > RS);
    f_valid = f_valid & (meno==1);
end

if strcmp(Clin, 'Low')
    f_valid = f_valid & (ClinicalRisk == 0);
elseif strcmp(Clin, 'High')
    f_valid = f_valid & (ClinicalRisk == 1);
end

f_Arm2 = f_valid & ismember(Arm,'B');
f_Arm3 = f_valid & ismember(Arm,'C');

%%
close all

draw_kaplan_benefit(times(f_Arm2), cens(f_Arm2), times(f_Arm3), cens(f_Arm3), prognosis_measure);

