import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import time

from model import Stock_rnn, Stock_cls
from torch.utils.data import DataLoader
from dataset import Stock_Train_Dataset, Stock_Val_Dataset


def parse_args(parser):
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('-s', '--seq_length', type=int, default=7,
                        help='몇일치를 보고 예측을 진행할것인지')
    parser.add_argument('--input_dim', type=int, default=5, help='sehl거래량이라 5')
    parser.add_argument('--hidden_dim', type=int, default=10, help='')#너무 작음
    parser.add_argument('--output_dim', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='')
    parser.add_argument('--epoch', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--day', type=int, default=2,
                        help='몇일치 주식 데이터를 가지고 올것인지')
    parser.add_argument('--cls_output', type=int, default=3)
    parser.add_argument('--checkpoint_path', type=str, default='./stock_cls.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


#종가의 리스트를 주면 변동성을 알려주는 함수
def stock_variability(close):
    gap = []

    for i in range(len(close) - 1):
        gap.append(np.log(close[i + 1] / close[i]))

    variance = np.var(gap)
    result = math.sqrt(variance * 252)
    return result

def validate(args):
    count = 0
    total = 0

    net = Stock_cls(args.input_dim, args.hidden_dim, args.cls_output, 1).to(args.device)

    val_set = Stock_Val_Dataset(args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              drop_last=True)

    net.load_state_dict(torch.load(args.checkpoint_path))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            testX, answers = batch
            predictions = net(testX) #.data.cpu().numpy()
            _, predictions = torch.max(predictions, 1)
            total += answers.size(0)
            for answer, prediction in zip(answers, predictions):
                print('answer : {}, prediction : {}'.format(answer, prediction))
                count += 1 if answer == prediction else 0
        print('accracy : ', count / total)

def main(args):
    start = time.time()

    torch.manual_seed(123)

    trainset = Stock_Train_Dataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                              drop_last=True)
    #net = Stock_rnn(args.input_dim, args.hidden_dim, args.output_dim, 1).to(device)
    cls_net = Stock_cls(args.input_dim, args.hidden_dim, args.cls_output, 1).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(cls_net.parameters(), lr=args.learning_rate)
    print('train start')
    cls_net.train()
    for i in range(args.epoch + 1):
        for k, batch in enumerate(train_loader):
            trainX_tensors, trainY_tensors = batch
            optimizer.zero_grad()
            outputs = cls_net(trainX_tensors)
            #print('outputs : ', outputs)
            #print('target : ', trainY_tensors.long().squeeze())
            loss = criterion(outputs, trainY_tensors.long().squeeze())
            loss.backward()
            optimizer.step()
        print(i, loss.item())
    #------------------------test-----------------------

    # for i, j in zip(testY, net(testX_tensor).data.cpu().numpy()):
    #     print('answer : {}, prediction : {}'.format(i, j))

    # plt.plot(testY)
    # plt.plot(net(testX_tensor).data.cpu().numpy())
    # plt.legend(['original', 'prediction1',])# 'prediction2', 'prediction3'])
    # plt.show()

    print('train finish')
    torch.save(cls_net.state_dict(), args.checkpoint_path)
    print('save check point')
    print('time :', time.time() - start)
    validate(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SoMa Stock Training And Test')
    parser = parse_args(parser)
    args = parser.parse_args()
    if args.mode == 'train':
        main(args)
    elif args.mode == 'val':
        validate(args)
