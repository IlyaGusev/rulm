import torch
from ignite.engine import Engine
from ignite.metrics import CategoricalAccuracy


def create_lm_trainer(model, optimizer, loss_fn, device=None, grad_clipping: int=5.):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        y_pred = model(batch)
        loss = loss_fn(y_pred, batch["y"])
        loss.backward()
        if grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_lm_evaluator(model, metrics=None, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            y_pred = model(batch)
            return y_pred, batch["y"]

    engine = Engine(_inference)
    if metrics:
        for name, metric in metrics.items():
            metric.attach(engine, name)
    return engine


class MaskedCategoricalAccuracy(CategoricalAccuracy):
    def update(self, output):
        y_pred, y = output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y).view(-1)

        true_zeros = torch.eq(y, y.new_zeros(y.size())).view(-1)
        pred_zeros = torch.eq(indices, indices.new_zeros(indices.size())).view(-1)
        twos = true_zeros.new_full(true_zeros.size(), 2)

        num_correct_zeros = torch.sum(torch.eq(true_zeros + pred_zeros, twos)).item()
        self._num_examples += correct.shape[0] - torch.sum(true_zeros).item()
        self._num_correct += torch.sum(correct).item() - num_correct_zeros

