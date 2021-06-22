# coding: utf-8
from utils import *
import torch.nn.functional as F
from net import hashing_net
import torchvision
import torch.optim as optim
from datetime import datetime
import sys
import argparse
from loss import ClassWiseLoss
import torch.backends.cudnn as cudnn
import time
import torchvision.transforms.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch Implementation of Deep Class-Wise Hashing')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-p', '--path', type=str, help='model path')
parser.add_argument('-u', '--upd_grad', action='store_true', help='whether updating class centroids by gradient descent, default: False')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('--dataset', type=str, default='cifar100', help='which dataset for training: {cifar10, cifar100}')
parser.add_argument('--bs', type=int, default=128, help='Batch size of each iteration')
parser.add_argument('--network', type=str, default='res50',
                    help='imagenet pre-trained network to use: {googlenet, res50, vgg19}')
parser.add_argument('--len', type=int, default=48, help='binary codes length')
parser.add_argument('--inv_var', default=0.5, type=float, help='value of 1.0/ $\sigma^2$')
parser.add_argument('--weight_cubic', default=10, type=float, help='balance weight on cubic loss')
parser.add_argument('--weight_vertex', default=0.01, type=float, help='balance weight on vertex loss')

args = parser.parse_args()


EPOCHS = 150


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using: " + str(device))

cudnn.benchmark = True
Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_train = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize,
    ])

transform_test = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    Normalize,
])


if args.dataset == "cifar10":

    trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=False,
                                        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)


else:

    trainset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=True, download=False,
                                        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

classes = len(trainset.classes)
print("number of classes: ", classes)
print("number of training images: ", len(trainset))
print("number of test images: ", len(testset))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""

    lr = args.lr * (0.1 ** (epoch // 50))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr  # lr
    return lr


def centers_computing(model, data_loader, classes, length):

    U = []
    labels = []
    centers = torch.Tensor(classes, length).cuda()
    model.eval()
    for iter, (data, target, *_) in enumerate(data_loader):
        data_input, target = data.cuda(), target.cuda()
        output = model(data_input)
        U.append(output.data)
        labels.append(target)
    U = torch.cat(U).cuda()
    labels = torch.cat(labels).squeeze().cuda()
    for i in torch.unique(labels).tolist():
        index_list = torch.nonzero(labels == i).squeeze()
        centers[i, :] = U[index_list, :].sum(dim=0) / index_list.size(0)

    return centers


def train(length, save_path):

    print('==> Preparing training data..')
    net = hashing_net(args.network, length).create_model()
    net = torch.nn.DataParallel(net).to(device)
    centers = torch.randn(classes, length).cuda().detach()
    best_MAP = 0
    best_epoch = 1
    best_loss = 1e6
    print('==> Building model..')

    criterion_beacon = ClassWiseLoss(num_classes=classes, bit_length=length, update_grad=args.upd_grad, use_gpu=True)

    if args.upd_grad:

        optimizer = optim.SGD(net.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    else:

        optimizer = optim.SGD([
            {'params': net.module.parameters(), 'weight_decay': 5e-4},
            {'params':  criterion_beacon.parameters(), 'weight_decay': 5e-4}], lr=args.lr, momentum=0.9)


    since = time.time()
    for epoch in range(EPOCHS):

        print('==> Epoch: %d' % (epoch + 1))
        torch.autograd.set_detect_anomaly(True)
        net.train()
        adjust_learning_rate(optimizer, epoch)
        dcwh_loss = AverageMeter()

        for batch_idx, (inputs, targets, *_) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            features = net(inputs)
            if not args.upd_grad:
                loss_beacon = criterion_beacon(features, targets, centers)
            else:
                loss_beacon = criterion_beacon(features, targets)
            loss_cubic = F.relu(-1.1 - features).sum() + F.relu(features - 1.1).sum() / len(inputs)     # Stage I: Establishment with cubic constraint
            Bbatch = torch.sign(features)
            loss_l3 = (Bbatch - features).pow(2).sum() / len(inputs)    # Stage II: Refine with vertex constraint
            loss = loss_beacon + (args.weight_cubic * loss_cubic + args.weight_vertex * loss_l3) / len(inputs)
            dcwh_loss.update(loss.item(), len(inputs))
            loss.backward()
            optimizer.step()

        print("[epoch: %d]\t[hashing loss: %.3f ]" % (epoch+1, dcwh_loss.avg))
        if (epoch+1) % 2 == 0:  # Update centers periodically each two epochs
            net.eval()
            centers = centers_computing(net, train_loader, classes, length)

        if (epoch+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                trainB, train_labels = compute_result(train_loader, net, device)

                testB, test_labels = compute_result(test_loader, net, device)
                mAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
                if mAP > best_MAP:
                    best_MAP = mAP

            print("[epoch: %d]\t[hashing loss: %.3f\t [mAP: %.2f%%]" % (epoch+1, dcwh_loss.avg, float(mAP)*100.0))

        if dcwh_loss.avg < best_loss:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(net.state_dict(), './checkpoint/%s' %save_path)
            best_loss = dcwh_loss.avg
            best_epoch = epoch + 1

        if (epoch + 1 - best_epoch) > 50:
            print("Training terminated at epoch %d" %(epoch + 1))
            break

    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s with best mAP {:.2%}".format(time_elapsed // 60, time_elapsed % 60, float(best_MAP)))
    print("Model saved as %s" % save_path)


def test(length, load_path):
    assert os.path.exists(os.path.join("./checkpoint", load_path)), "model path not found!"
    checkpoint = torch.load("./checkpoint/%s" % load_path)
    net = hashing_net(args.network, length).create_model()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint)

    net.eval()
    with torch.no_grad():

        since = time.time()
        trainB, train_labels = compute_result(train_loader, net, device)
        testB, test_labels = compute_result(test_loader, net, device)
        MAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
        time_elapsed = time.time() - since
        print("Calculate mAP in {:.0f} min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print('[Evaluate Phase] MAP: %.2f%%' % (100. * float(MAP)))


if __name__ == '__main__':

    if not os.path.isdir('log_dcwh'):
        os.mkdir('log_dcwh')

    save_dir = './log_dcwh'

    if args.evaluate:
        test(args.len, args.path)

    else:

        sys.stdout = Logger(os.path.join(save_dir, str(args.len) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
        print("[Configuration] Training on dataset: %s\n len: %d \n Batch_size: %d\n learning rate: %.3f\n #Epoch: %d\n " %(args.dataset, args.len, args.bs, args.lr, EPOCHS))

        print("HyperParams:\ninv_var: %.3f\t weight_cubic: %.3f\t weight_vertex: %.3f" % (args.inv_var, args.weight_cubic, args.weight_vertex))
        train(args.len, args.path)
