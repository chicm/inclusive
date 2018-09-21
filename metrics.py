import torch

def accuracy(logits, target):
    '''
    logits: N,C
    target: N,C
    '''
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []
    target = target.byte()
    #pred = pred.topk(3, )
    for t in thresholds:
        pred = (torch.sigmoid(logits) > t).byte()
        corrects = (pred.eq(target) * target).sum().item()
        incorrects = (pred.ne(target) * (target.eq(0))).sum().item()
        results.append((corrects, incorrects, corrects/(incorrects+1), t))
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