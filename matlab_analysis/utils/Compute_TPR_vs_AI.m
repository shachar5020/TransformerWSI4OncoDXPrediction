%{
    label - a vector of nx1 with true RS values
    AI_score - a vector of nx1 with predicted RS values
    useBoot - true for bootstrapping
    
    For each binarizing threshold of AI score, computes TPR 
%}


function TPR_vec = Compute_TPR_vs_AI(label, AI_score, useBoot)

if ~useBoot
    [~, s_i] = sort(AI_score);    
    label_s = label(s_i);
    
    TPR_vec = 1-cumsum(label_s == 1)/sum(label == 1);
    
else
    boot_N = 1000;
    n = length(AI_score);
    TPR_vec_boot = zeros(n, boot_N);

    for i = 1:boot_N  
        r = randi(n,n,1);            
        label_r = label(r);
        AI_score_r = AI_score(r);

        [~, s_i] = sort(AI_score_r);    
        label_s = label_r(s_i);
        
        TPR_vec_boot(:,i) = 1-cumsum(label_s == 1)/sum(label_r == 1);
    end

    CI = 0.95;
    TPR_vec = ...
        [mean(TPR_vec_boot,2)...
    quantile(TPR_vec_boot, 0.5 - CI/2, 2)...
    quantile(TPR_vec_boot, 0.5 + CI/2, 2)];

end



