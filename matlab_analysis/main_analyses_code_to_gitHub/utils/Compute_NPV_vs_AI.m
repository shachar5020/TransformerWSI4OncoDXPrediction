%{
    label - a vector of nx1 with true RS values
    AI_score - a vector of nx1 with predicted RS values
    useBoot - true for bootstrapping
    
    For each binarizing threshold of AI score, computes NPV
%}


function NPV_vec = Compute_NPV_vs_AI(label, AI_score, useBoot)

if ~useBoot
    [~, s_i] = sort(AI_score);    
    label_s = label(s_i);
    
    label_s_0 = (label_s == 0);
    label_s_01 = (label_s == 0) + (label_s == 1);
    
    NPV_vec = cumsum(label_s_0)./cumsum(label_s_01);
else
    boot_N = 1000;
    n = length(AI_score);
    NPV_vec_boot = zeros(n, boot_N);

    for i = 1:boot_N  
        r = randi(n,n,1);            
        label_r = label(r);
        AI_score_r = AI_score(r);

        [~, s_i] = sort(AI_score_r);    
        label_s = label_r(s_i);
    
        label_s_0 = (label_s == 0);
        label_s_01 = (label_s == 0) + (label_s == 1);
        
        NPV_vec_boot(:,i) = cumsum(label_s_0)./cumsum(label_s_01);

    end

    CI = 0.95;
    NPV_vec = ...
        [mean(NPV_vec_boot,2)...
    quantile(NPV_vec_boot, 0.5 - CI/2, 2)...
    quantile(NPV_vec_boot, 0.5 + CI/2, 2)];
end