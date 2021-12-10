from transformers import BertForSequenceClassification
from dataloader import TokenDataset
from functools import partial
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import math
import logging

logging.basicConfig(level=logging.ERROR)


def KFoldCV(config, checkpoint_dir = None, data_dir = None):
    fold_var = 1
    cm_holder = []
    # read csv
    rslt_df = pd.read_csv(data_dir)
    # kfold parameters
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    for train_index, test_index in kf.split(rslt_df):
        print('this is fold var: ' + str(fold_var))
        # generate splitted dataframe
        train_df = rslt_df.iloc[train_index]
        val_df = rslt_df.iloc[test_index]
        # build dataloaders
        train_data = TokenDataset(train_df, 'clean_text','race')
        val_data = TokenDataset(val_df, 'clean_text', 'race')
        train_counts=pd.value_counts(train_df['race']).tolist()
        print(train_counts)
        train_weights= 1./ torch.tensor(train_counts, dtype=torch.float)
        train_targets = train_data.getLabels()
        train_samples_weights = train_weights[train_targets]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_samples_weights, num_samples=len(train_samples_weights), replacement=True)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size = int(config["batch_size"]), shuffle = False, sampler=train_sampler)
        valloader = torch.utils.data.DataLoader(val_data, batch_size = 1,shuffle=False)
        # initialize model, optimizer
        bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 4, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                bert = nn.DataParallel(bert)
        bert.to(device)

        #criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(bert.parameters(),lr=config["lr"], eps = 1e-8)
        
        # Total number of training steps
        total_steps = len(trainloader) * 5

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)


        for epoch in range(5):
            print('this is epoch number ' + str(epoch)+'\n')
            running_loss = 0.0
            epoch_steps = 0
            bert.train(True)
            for i, (input_ids,attention_mask,label) in enumerate(trainloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = bert(input_ids,attention_mask=attention_mask,labels=label)

                loss = output[0]
                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
                loss.sum().backward()
                running_loss += loss.sum().item()
                
                

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # print statistics
                
                epoch_steps += 1
                if i % 20 == 19:  # print every 20 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / 20))
                    running_loss = 0.0

        y_pred, y_true = [], []
        total = 0
        correct = 0
        val_loss=0.
        val_steps = 0
        bert.eval()
        with torch.no_grad():
            for idx, (input_ids,attention_mask,label) in enumerate(valloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                
                output = bert(input_ids,attention_mask=attention_mask,labels=label)
                loss = output[0]
                logits=output[1]
                
                val_loss += loss.cpu().numpy()
                pred=logits.argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                y_pred.append(pred)
                y_true.append(label)
                val_steps += 1
        y_pred = torch.cat(y_pred).cpu().detach().numpy()
        y_true = torch.cat(y_true).cpu().detach().numpy()
        cm = confusion_matrix(y_true, y_pred)
        print('confusion matrix fold ' + str(fold_var))
        print(cm)
        cm_holder.append(cm)
        print(classification_report(y_true, y_pred, target_names = ['Black', 'Hispanic/Latino','Asian','White']))
        print("Accuracy: ", score(y_true, y_pred))
        print("test loss: {}".format(val_loss/ val_steps))

        with tune.checkpoint_dir(fold_var) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "fold-checkpoint")
            torch.save(bert.state_dict(), path)
        
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        print('end of fold var:' + str(fold_var))
        fold_var += 1
        
        
    cm_sum = np.zeros((4,4))
    for each in cm_holder:
        cm_sum += each
    cm_sum = cm_sum.astype('float') / cm_sum.sum(axis=1)[:, np.newaxis]
    print('final averaged confusion matrix: ')
    print(cm_sum)
    print("\nFinished K Fold CV\n")



def main(num_samples=10, max_num_epochs=4, gpus_per_trial=2):
    
    config = {
        "lr": tune.choice([5e-5, 3e-5, 1e-4]),
        "batch_size": tune.choice([12,16,24])
    }
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(KFoldCV, checkpoint_dir = None, data_dir = '/projectnb/cs640g/students/yliao127/tweets_data_w_race/merged_tweets_per_user.csv'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        keep_checkpoints_num=1, 
        local_dir = '/projectnb/cs640g/students/yliao127/checkpoint_dir',
        checkpoint_score_attr="accuracy",
        num_samples=num_samples,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("mean_accuracy", "max")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))



if __name__ == "__main__":
    main(num_samples= 6, gpus_per_trial=2)