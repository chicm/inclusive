from sklearn import metrics
import torch
import numpy as np


def accuracy(logits, target):
    '''
    logits: N,C
    target: N,C
    '''
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []
    target = target.byte()
    #pred = pred.topk(3, )
    for t in thresholds:
        pred = (torch.sigmoid(logits) > t).byte()
        corrects = (pred.eq(target) * target).sum().item()
        incorrects = (pred.ne(target) * (target.eq(0))).sum().item()
        results.append((corrects, incorrects, round(corrects/(incorrects+1),3), t, target.sum().item()))
    return results

def topk_accuracy(output, label, topk=(1,100)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res

def f2_scores(logits, target):
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    preds = torch.sigmoid(logits)
    results = []
    for t in thresholds:
        score = round(f2_score(target, preds, t).item(), 4)
        results.append((t, score))
    return results

def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

if __name__ == '__main__':
    y_pred = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    py_pred = torch.from_numpy(y_pred)
    py_true = torch.from_numpy(y_true)

    fbeta_pytorch = f2_score(py_true, py_pred)
    #fbeta_sklearn = metrics.fbeta_score(y_true, y_pred, 2, average='samples')
    #print('Scores are {:.3f} (sklearn) and {:.3f} (pytorch)'.format(fbeta_sklearn, fbeta_pytorch))
    print(fbeta_pytorch.cuda().item())
    print('Scores are {:.3f}'.format(fbeta_pytorch))