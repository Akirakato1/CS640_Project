from sklearn import metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def test(net, test_loader, device):
    y_pred, y_true = [], []
    losses = 0.
    avgloss = 0.
    with torch.no_grad():
        for idx, (input_ids, attention_mask, label) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            output = net(input_ids, attention_mask=attention_mask, labels=label)
            loss = output[0]
            logits = output[1]
            losses += loss.item()
            avgloss = losses / (idx + 1)
            pred = logits.argmax(dim=-1)
            y_pred.append(pred)
            y_true.append(label)
    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    print(metrics.classification_report(y_true, y_pred))
    print("Accuracy: ", metrics.accuracy_score(y_true, y_pred))
    print("test loss: {}".format(avgloss))
    return avgloss


def train(net, train_loader, test_loader, optimizer, scheduler, device, epochs):
    losslist = []
    tlosses = []
    for epoch in range(epochs):
        losses = 0
        for idx, (input_ids, attention_mask, label) in enumerate(train_loader):
            net.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            output = net(input_ids, attention_mask=attention_mask, labels=label)

            loss = output[0]
            losslist.append(loss.item())
            losses += loss.item()
            loss.backward()
            #             torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            if (idx + 1) % 20 == 0:
                print("epoch:{}, step:{}, loss:{}, lr:{}".format(epoch + 1, idx + 1, losses / 20,
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
                losses = 0
        #         test(net, train_loader, bert, tokenizer, criterion, device)
        tloss = test(net, test_loader, device)
        tlosses.append(tloss)
        scheduler.step()

        if (epoch + 1) % 1 == 0:
            model_path = "./model/bert_ft_{}.model".format(epoch + 1)
            torch.save(net, model_path)
            print("saved model: ", model_path)
    plt.subplot(2, 1, 1)
    plt.plot(losslist)
    plt.subplot(2, 1, 2)
    plt.plot(tlosses)
    plt.show()
