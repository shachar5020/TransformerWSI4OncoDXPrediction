%{
    Arm - nx1 vector, with the arm of TAILORx for each patient
    times - nx1 vector with patient survival times
    cens - nx1 vector with patient cesoring indicators (0 is event)
    prognosis measure - needed for correcting HR
    prob_adj - scalar, how many patients to sample from RS<16
    

    outputs: adjusted times and censored (nx1 vectors) by adding a sample
    of the patients from RS>=26 to those with 11<=RS<26
%}


function [times_adj_Chemo, cens_adj_Chemo, times_adj_NoChemo, cens_adj_NoChemo] = get_adjusted_times_cens_postmeno(Arm, times, cens, prognosis_measure, prob_adj)


%% randomly split Arm D to two: half of the patients go to arm B, and half to arm C
n = length(Arm);
r_split = rand(n, 1);
f_split_1 = r_split >= 0.5;
f_split_2 = r_split < 0.5;

f_Arm2 = ismember(Arm,'B');
f_Arm3 = ismember(Arm,'C');
f_Arm4_to_Arm2 = ismember(Arm,'D') & f_split_1;
f_Arm4_to_Arm3 = ismember(Arm,'D') & f_split_2;

%% chemo group: Arm 3 + sample from Arm 4 group chemo
times_Arm4_to_Arm3 = times(f_Arm4_to_Arm3);
cens_Arm4_to_Arm3 = cens(f_Arm4_to_Arm3);

r_samp_Arm4_to_Arm3 = rand(size(times_Arm4_to_Arm3));
f_samp = r_samp_Arm4_to_Arm3<prob_adj;

times_adj_Chemo = [times(f_Arm3); times_Arm4_to_Arm3(f_samp)];
cens_adj_Chemo = [cens(f_Arm3);cens_Arm4_to_Arm3(f_samp)];

%% no chemo group: Arm 2 + samples from Arm 4 group chemo corrected to no chemo
% compute corrected times and cens to Arm4_to_Arm2 group: 

% survival curve
[KM_time_Arm4_to_Arm2_chemo, KM_prob_Arm4_to_Arm2_chemo] = get_kaplan_meier_curve(times(f_Arm4_to_Arm2), cens(f_Arm4_to_Arm2));

if strcmp(prognosis_measure, 'distant-recurrence-free')
    HR_correction = 0.44;
elseif strcmp(prognosis_measure, 'recurrence-free')
    HR_correction = 0.44;
elseif strcmp(prognosis_measure, 'disease-free')
    HR_correction = 0.45;
end


% correction for the chemo group
KM_time_Arm4_to_Arm2_no_chemo = KM_time_Arm4_to_Arm2_chemo;
KM_prob_Arm4_to_Arm2_no_chemo = KM_prob_Arm4_to_Arm2_chemo.^(1/HR_correction(1));

N_no_chemo = sum(f_Arm4_to_Arm2);
[times_Arm4_to_Arm2, cens_Arm4_to_Arm2] = event_times_from_survival(KM_time_Arm4_to_Arm2_no_chemo, KM_prob_Arm4_to_Arm2_no_chemo, N_no_chemo);

r_samp_Arm4_to_Arm2 = rand(size(times_Arm4_to_Arm2));
f_samp = r_samp_Arm4_to_Arm2<prob_adj;

times_adj_NoChemo = [times(f_Arm2); times_Arm4_to_Arm2(f_samp)];
cens_adj_NoChemo = [cens(f_Arm2);cens_Arm4_to_Arm2(f_samp)];
