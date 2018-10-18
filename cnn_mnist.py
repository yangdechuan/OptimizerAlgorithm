import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tensorboard_logger import configure, log_value

# global step
step = 0


class Net(nn.Module):
    """Two layer CNN and two layer fc Model."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # the shape of x: (batch_size, 20, 4, 4)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        global step
        log_value("training loss", loss.item(), step=step)
        step += 1


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description="Mnist CNN Demo")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[
            "sgd",
            "sgd_with_momentum",
            "sgd_with_momentum_and_nesterov",
            "agagrad",
            "rmsprop",
            "adadelta",
            "adam"],
        default="sgd")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    if "sgd" in args.optimizer:
        configure(
            os.path.join(
                "log",
                args.optimizer +
                "_lr{}".format(
                    args.lr)))
    else:
        configure(os.path.join("log", args.optimizer))

    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data", train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3801,))
                                   ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data", train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3801,))
                                   ])),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    model = Net().to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd_with_momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum)
    elif args.optimizer == "sgd_with_momentum_and_nesterov":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters())
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters())
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters())
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters())
    else:
        print("Error: Unknown optimizer!")

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == "__main__":
    main()
