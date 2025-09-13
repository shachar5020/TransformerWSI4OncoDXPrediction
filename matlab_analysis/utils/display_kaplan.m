%{
    times - a vector of nx1 with survival times
    cens - a corresponding vector of nx1. 0 means event
    RS - the true RS values
    AI_score = the predicted RS values
    th - the stratification threshold
    str_endpoint - text for the ylabel

    Computes HR, P value, and displays Kaplan Meier plot
%}

function display_kaplan(times, cens, AI_score, RS, th, str_endpoint)

n = length(AI_score);
%% get curves


f1 = AI_score < th;
f2 = AI_score >= th;
f3 = RS < th;
f4 = RS >= th;

[time_s_1, prob_vec_1] = get_kaplan_meier_curve(times(f1), cens(f1));
[time_s_2, prob_vec_2] = get_kaplan_meier_curve(times(f2), cens(f2));
[time_s_3, prob_vec_3] = get_kaplan_meier_curve(times(f3), cens(f3));
[time_s_4, prob_vec_4] = get_kaplan_meier_curve(times(f4), cens(f4));

%% compute HR and P value
times_AI = [times(f2); times(f1)];
cens_AI = [cens(f2); cens(f1)];
group_AI = [2*ones(sum(f2),1); ones(sum(f1),1)];

[b, ~, ~, stats] = coxphfit(group_AI, times_AI, 'Censoring', cens_AI);
crit = 1.96; % 95% CI
stats_AI.p = stats.p;
stats_AI.HR = [exp(b) exp(b - crit * stats.se) exp(b + crit * stats.se)];
stats_AI.HR = stats_AI.HR(1);


times_RS = [times(f4); times(f3)];
cens_RS = [cens(f4); cens(f3)];
group_RS = [2*ones(sum(f4),1); ones(sum(f3),1)];

[b, ~, ~, stats] = coxphfit(group_RS, times_RS, 'Censoring', cens_RS);
crit = 1.96;  % 95% CI
stats_RS.p = stats.p;
stats_RS.HR = [exp(b) exp(b - crit * stats.se) exp(b + crit * stats.se)];
stats_RS.HR = stats_RS.HR(1);


%% display figure
figure;
hold on;
stairs(time_s_1, prob_vec_1(:, 1), 'color', 'r', 'LineWidth',2.5)
stairs(time_s_2, prob_vec_2(:, 1), 'color', 'b', 'LineWidth',2.5)
stairs(time_s_3, prob_vec_3(:, 1), '--', 'color', 'r', 'LineWidth',2.5)
stairs(time_s_4, prob_vec_4(:, 1), '--', 'color', 'b', 'LineWidth',2.5)

ylim_low = 0.5;
ylim([ylim_low 1])
xlim([0 15])

% legend(sprintf('AI score < %d (%d)',th_AI, sum(f1)), sprintf('AI score ≥ %d (%d)',th_AI, sum(f2)), sprintf('RS < %d (%d)',th_AI,sum(f3)), sprintf('RS ≥ %d (%d)',th_AI,sum(f4)), 'Location','southeast')
set(gca, 'fontsize', 17)
legend(...
    sprintf('AI Low-Risk: %d (%.0f%%)', sum(f1), sum(f1)/n*100), ...
    sprintf('AI High-Risk: %d (%.0f%%)', sum(f2), sum(f2)/n*100), ...
    sprintf('RS Low-Risk: %d (%.0f%%)', sum(f3), sum(f3)/n*100), ...
    sprintf('RS High-Risk: %d (%.0f%%)', sum(f4), sum(f4)/n*100), ...
    'Location','southeast', 'fontsize', 15);
hold off;
grid on;
ylabel(str_endpoint, 'FontSize',17)
xlabel('Years')



if stats_RS.p < 0.001
    P = 'P<0.001';
elseif stats_RS.p < 0.01
    P = 'P<0.01';
else
    P = sprintf('P=%.3f', stats_RS.p);
end
P_str_RS = sprintf('%s, HR=%.1f',P, stats_RS.HR);


if stats_AI.p < 0.001
    P = 'P<0.001';
elseif stats_AI.p < 0.01
    P = 'P<0.01';
else
    P = sprintf('P=%.3f', stats_AI.p);
end
P_str_AI = sprintf('%s, HR=%.1f',P, stats_AI.HR);

text(0.3, ylim_low + 0.11*(1-ylim_low), 'RS:', 'FontSize',15)
text(0.3, ylim_low + 0.04*(1-ylim_low), 'AI:', 'FontSize',15)

text(1.4, ylim_low + 0.11*(1-ylim_low), P_str_RS, 'FontSize',15)
text(1.4, ylim_low + 0.04*(1-ylim_low), P_str_AI, 'FontSize',15)



