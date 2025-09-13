close all
clear all;
clc

%%
T_per_patient = readtable('metadata/TAILORx_metadata.xlsx');
n = size(T_per_patient,1);

restoredefaultpath
addpath('utils/')
%% clinical variables

Arm = T_per_patient.rxarm;
InAnalysis = T_per_patient.InAnalysis;
RS = T_per_patient.RS;
meno = ismember(T_per_patient.meno, 'Post');
AI_score = T_per_patient.AI_score_combined;

%%
prognosis_measure = 'disease-free';
[times, cens] = getTailorPrg(T_per_patient, prognosis_measure);

f_valid_1 = ~isnan(times) & ~isnan(cens) & ~isnan(AI_score);
f_valid_1 = f_valid_1 & (InAnalysis == true);
f_valid = f_valid_1 & (meno==0);

%% For AI low risk (AI<16), compute proportion of patients with RS<16 out of RS<26

prop = sum(f_valid & (AI_score >= 26) & (RS<16))/sum(f_valid & (AI_score >= 26) & (RS<26));

%% compute the proportion of patients we need to sample from RS<16 in order to get prop

%{
    prop: the desired portion of patients below 16 out of all patients
    prop_adj: the portion of patients we should sample from the patients below 16
    
    given prop, find prop_adj such that:
    a = P(RS<16)
    b = P(16<=RS<26)
    prop = a*prop_adj/(a*prop_adj + b)
    -> 
    prop_adj = prop/(1-prop)*b/a
%}

a = sum(f_valid & (RS<16));
b = sum(f_valid & (RS>=16) & (RS<26));

prop_adj = prop/(1-prop)*b/a; % proportion of patients to sample from RS>=26 for the adjustment
%% get the adjusted times and censoring
[times_Chemo, cens_Chemo, times_NoChemo, cens_NoChemo] = get_adjusted_times_cens_premeno(Arm(f_valid), times(f_valid), cens(f_valid), prop_adj, RS(f_valid));
 
%% display KM plot
draw_kaplan_benefit(times_NoChemo, cens_NoChemo, times_Chemo, cens_Chemo, prognosis_measure);

