%{
    Arm - nx1 vector, with the arm of TAILORx for each patient
    times - nx1 vector with patient survival times
    cens - nx1 vector with patient cesoring indicators (0 is event)
    prob_adj - scalar, how many patients to sample from RS<16
    RS - nx1 vector with the true RS values

    outputs: adjusted times and censored (nx1 vectors) by adding a sample
    of the patients from RS<16 to those with 16<=RS<26
%}


function [times_adj_Chemo, cens_adj_Chemo, times_adj_NoChemo, cens_adj_NoChemo] = get_adjusted_times_cens_premeno(Arm, times, cens, prob_adj, RS)

n = length(Arm);
r_split = rand(n, 1);
f_split_1 = r_split >= 0.5;
f_split_2 = r_split < 0.5;

r_samp = rand(n, 1);
f_samp = r_samp<prob_adj;

f_RS_11_16 = (RS >= 11) & (RS < 16);
f_RS_11_16_B = f_RS_11_16 & ismember(Arm,'B');
f_RS_11_16_C = f_RS_11_16 & ismember(Arm,'C');

f_RS_lower_11 = (RS < 11);
f_RS_lower_11_split_B = f_RS_lower_11&f_split_1;
f_RS_lower_11_split_C = f_RS_lower_11&f_split_2;
f_goes_to_Arm_B = (f_RS_11_16_B | f_RS_lower_11_split_B) & f_samp;
f_goes_to_Arm_C = (f_RS_11_16_C | f_RS_lower_11_split_C) & f_samp;

f_noChemo_baseline = (RS>=16) & ismember(Arm,'B');
f_Chemo_baseline = (RS>=16) & ismember(Arm,'C');

f_noChemo = f_noChemo_baseline | f_goes_to_Arm_B;
f_Chemo = f_Chemo_baseline | f_goes_to_Arm_C;

times_adj_NoChemo = times(f_noChemo);
cens_adj_NoChemo = cens(f_noChemo);
times_adj_Chemo = times(f_Chemo);
cens_adj_Chemo = cens(f_Chemo);

