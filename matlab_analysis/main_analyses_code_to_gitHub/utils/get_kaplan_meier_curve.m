%{
    time - a vector of nx1 times
    censored - a corresponding vector of nx1. 0 means not censored (real
    event).
    returnes the x axis (sorted times) and y axis (probability for event)
    of the kaplan meier curve.

%}
%%
function [time_s, prob_vec] = get_kaplan_meier_curve(time, censored)

n = length(time);
[time_s, s_i] = sort(time);
censored_s = censored(s_i);
prob_vec = zeros(n,1);
p_curr = 1;
num = n;

for i =1:n
   if ~censored_s(i)
       p_curr = p_curr*(num-1)/num;
   end
   num = num - 1;
   prob_vec(i) = p_curr;
end

%% complete curve to (0,0)

time_s = [0; time_s(1); time_s];
prob_vec = [1; 1; prob_vec];

