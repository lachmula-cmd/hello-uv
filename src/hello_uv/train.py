import torch as nn
from hello_uv.fashion import make_loader

def accuracy(logits,y):
    max = logits.argmax(dim=1)
    val = (max == y).float()
    mean = val.mean()
    return mean.item()

def train_one_epoch(model, loader, loss_fn, opt):
    model.train()

    losses = []
    for x,y in loader:
        opt.zero_grad()
        scores = model(x)
        loss = loss_fn(scores,y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses.mean()

def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    correct_sum = 0
    total_examples = 0
    for x,y in loader:
        score = model(x)
        loss = loss_fn(score,y)
        losses.append(loss.item())
        correct = (x == y).float()
        correct_sum = correct.sum()
        total_examples = len(y)
    return (losses.mean(), correct_sum/total_examples)


if __name__ == "__main__":
    nn.manual_seed(42)

    make_loader("FashionMNIST",64, shuffle=True )

    make_loader("FashionMNIST",1000, shuffle=False)

    model = nn.sequential(
        nn.flatten(),
        nn.linear(784,10),
        nn.CrossEntropyLoss(),
        

    )








