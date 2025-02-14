import torch
import torcheval.metrics.functional as MF
def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse
    

def masked_fre_mae(preds, labels, null_val=None):
    # print(preds.shape, labels.shape)
    preds = torch.fft.rfft(preds, dim=1)
    labels = torch.fft.rfft(labels, dim=1)
    # print(preds.shape, labels.shape)
    null_val = labels.abs().min() if labels.abs().min() < 1 else torch.tensor(0)
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_f1_score(preds, labels, null_val=None, threshold_1=35, threshold_2=75):
    '''if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)'''
    
    preds_mean = torch.mean(preds, dim=1).reshape(-1)
    labels_mean = torch.mean(labels, dim=1).reshape(-1)
    
    preds_mean[preds_mean < threshold_1] = 0
    preds_mean[(threshold_1 <= preds_mean) & (preds_mean < threshold_2)] = 1
    preds_mean[threshold_2 <= preds_mean] = 2

    labels_mean[labels_mean < threshold_1] = 0
    labels_mean[(threshold_1 <= labels_mean) & (labels_mean < threshold_2)] = 1
    labels_mean[threshold_2 <= labels_mean] = 2


    loss = MF.multiclass_f1_score(preds_mean.long(), labels_mean.long(), num_classes=3, average=None)
    return loss