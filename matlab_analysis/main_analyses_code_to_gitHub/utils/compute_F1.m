function F1 = compute_F1(label_pred, label_true)

TPR = sum(label_pred == 1 & label_true == 1)/sum(label_true == 1);
PPV = sum(label_pred == 1 & label_true == 1)/sum(label_pred == 1);

F1 = 2*TPR*PPV/(TPR+PPV);