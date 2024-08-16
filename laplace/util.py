import torch
from tqdm import tqdm


@torch.no_grad()
def predict(test_loader, model, n_samples=20, apply_softmax=True, return_targets=False, delta=1, n_data=None):
    py = []
    targets = []
    count = 0

    for x, y in test_loader:
        if n_data is not None and count >= n_data:
            break

        x, y = delta*x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)
        count += len(x)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)

@torch.no_grad()
def predict_misclf(test_loader, model, n_samples=20, apply_softmax=True, return_targets=False, delta=1, n_data=None):
    py = []
    targets_correct = []
    targets_error = []
    count = 0

    for x, y_correct, y_error in test_loader:
        if n_data is not None and count >= n_data:
            break

        x, y_correct, y_error = delta*x.cuda(), y_correct.cuda(), y_error.cuda()
        targets_correct.append(y_correct)
        targets_error.append(y_error)
        
        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)
        count += len(x)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets_correct, dim=0), torch.cat(targets_correct, dim=0),
    else:
        return torch.cat(py, dim=0)
