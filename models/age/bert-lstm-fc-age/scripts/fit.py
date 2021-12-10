from sklearn import metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def test(net, test_loader, bert, criterion, device, batch_size):
    y_pred, y_true = [], []
    losses = 0.
    avgloss = 0.
    h = net.init_hidden(batch_size)
    with torch.no_grad():
        for idx, (input_ids, attention_mask, label) in enumerate(test_loader):
            h = tuple([each.data for each in h])
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if str(criterion) == 'CrossEntropyLoss()':
                label = label.to(device)
            elif str(criterion) in ['BCELoss()', 'BCEFocalLoss()']:
                label = label.float().to(device)
            else:
                print("Criterion type error\n")
                break
            #             last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            #             bert_output = last_hidden_states[0][:, 0]

            bert_output = bert(input_ids)[0]
            output = net(bert_output, h)
            if str(criterion) in ['BCELoss()', 'BCEFocalLoss()']:
                output = output.view(-1)
            losses += criterion(output, label)
            avgloss = losses / (idx + 1)
            y_pred.append(output)
            y_true.append(label)
    if str(criterion) == 'CrossEntropyLoss()':
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        print(criterion(y_pred, y_true))
        y_pred = np.argmax(y_pred.cpu(), axis=1)
        #     print(y_pred)
        print(metrics.classification_report(y_true.cpu(), y_pred))
    elif str(criterion) in ['BCELoss()', 'BCEFocalLoss()']:
        y_prob = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        y_pred = y_prob.cpu()
        y_true = y_true.cpu()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        print(metrics.classification_report(y_true, y_pred))
        acc = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy: ", acc)
        print("AUC: ", metrics.roc_auc_score(y_true, y_prob.cpu()))
    else:
        print("Criterion type error\n")
    print("test loss: {}".format(avgloss))
    return avgloss, acc


def train(net, train_loader, test_loader, bert, optimizer, criterion, scheduler, device, epochs, batch_size):
    losslist = []
    tlosses = []
    accs = []
    for epoch in range(epochs):
        losses = 0
        h = net.init_hidden(batch_size)
        for idx, (input_ids, attention_mask, label) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if str(criterion) == 'CrossEntropyLoss()':
                label = label.to(device)
            elif str(criterion) in ['BCELoss()', 'BCEFocalLoss()']:
                label = label.float().to(device)
            else:
                print("Criterion type error\n")
                break
            with torch.no_grad():
                bert_output = bert(input_ids)[0]
            #                 last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            #                 bert_output = last_hidden_states[0][:, 0]

            h = tuple([each.data for each in h])

            net.zero_grad()
            output = net(bert_output, h)
            if str(criterion) in ['BCELoss()', 'BCEFocalLoss()']:
                output = output.view(-1)
            loss = criterion(output, label)
            losslist.append(loss)
            losses += loss
            loss.backward()
            optimizer.step()
            if (idx + 1) % 2 == 0:
                print("epoch:{}, step:{}, loss:{}, lr:{}".format(epoch + 1, idx + 1, losses / 2,
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
                losses = 0
        #         test(net, train_loader, bert, tokenizer, criterion, device)
        tloss, acc = test(net, test_loader, bert, criterion, device, batch_size)
        tlosses.append(tloss)
        accs.append(acc)
        scheduler.step()

        if (epoch + 1) % 2 == 0:
            model_path = "./model/bert_lstm_{}.model".format(epoch + 1)
            torch.save(net, model_path)
            print("saved model: ", model_path)
    plt.subplot(2, 1, 1)
    plt.plot(losslist)
    plt.subplot(2, 1, 2)
    plt.plot(tlosses)
    plt.plot(accs)
    plt.show()
