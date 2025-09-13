
%{
    times_noChemo - survival time per patient for patients with no chemo
    cens_noChemo - censoring per patient for patients with no chemo (0 =
    event)
    times_Chemo - survival time per patient for patients with chemo
    cens_Chemo - censoring per patient for patients with chemo (0 =
    event)
    ylabel_str - text for Y label (endpoint)

    computes HR, P value, and displays the KM plot
%}

function draw_kaplan_benefit(times_noChemo, cens_noChemo, times_Chemo, cens_Chemo, ylabel_str)


[time_s_1, prob_vec_1] = get_kaplan_meier_curve(times_noChemo, cens_noChemo);
[time_s_2, prob_vec_2] = get_kaplan_meier_curve(times_Chemo, cens_Chemo);

figure;
hold on;
stairs(time_s_1, prob_vec_1(:, 1), 'color', 'r', 'LineWidth',2.5)
stairs(time_s_2, prob_vec_2(:, 1), 'color', 'b', 'LineWidth',2.5)
hold off;
ylim([0.5 1])
xlim([0 11])
n_1 = length(times_noChemo);
n_2 = length(times_Chemo);
grid on;
set(gca, 'fontsize', 17)
legend(...
    sprintf('%s (%d)', 'No Chemo', n_1), ...
    sprintf('%s (%d)', 'Chemo', n_2), ...
    'Location','southeast', 'fontsize', 15);

ylabel(ylabel_str, 'FontSize',17)
xlabel('Years')


%% display HR and P value
times_benefit = [times_Chemo; times_noChemo];
cens_benefit = [cens_Chemo; cens_noChemo];
group_benefit = [2*ones(length(times_Chemo),1); ones(length(times_noChemo),1)];

[b, ~, ~, stats] = coxphfit(group_benefit, times_benefit, 'Censoring', cens_benefit);
crit = 1.96;
results.p = stats.p;
HR = [exp(b) exp(b - crit * stats.se) exp(b + crit * stats.se)];

if results.p < 0.001
    P = 'P<0.001';
elseif results.p < 0.01
    P = 'P<0.01';
else
    P = sprintf('P=%.3f', results.p);
end

P_str = sprintf('%s, HR=%.2f (%.2f–%.2f)',P, HR(1), HR(2), HR(3));
text(0.5, 0.5 + 0.02, P_str, 'FontSize',15)
