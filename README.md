# CS640_Project
## Goal
Demographic (Age (<21 and >=21) and Race (black, white, Hispanic/Latino, Asian) prediction of Twitter users
## Existing Data
Twitter user profile and it's most recent 100 tweets
## Process
1. Text Processing
2. Train CNN model on UTK face
3. Race prediction:<br />
(1) CNN<br />
(2) TF-IDF<br />
(3) BERT<br />
4. Age prediction:<br />
(1) TF-IDF<br />
(2) Navie bayes<br />
(3) Bert<br />
(4) Bert-LSTM-FC<br />
## Models
Models directory contains all final models tried in their own folders.
### Tf-idf Vectorizer Logistic Regression
We were motivated to try this approach as a baseline because the project description did provide an attempted approach of Tf-idf logistic regression. 
##### Model description:
The model vectorizes each tweet as a document and only keeps features that have a minimum data frequency of 4 to reduce overfitting. Feature space around 10k. 

The data distribution over users groupby race is {Black: 374, Hispanic: 241, Asian: 140, White: 3184}, which is heavily imbalanced towards White. We trained a mode with undersampling and one without over 5 fold to see the precision recall scores. 
### BERT
We choose Bert because it’s already pre-trained on a lot of text and we are expecting Bert to be able to extract some feature that is related to user race.
##### Model Description:
The model is BertForSequenceClassification.from_pretrained("bert-based-uncased"), we set the parameters of the model with num_labels = 4, output_attentions = False, output_hidden_states = False.

We performed hyperparameter tuning by using the Ray Tune package, and the hyperparameters we tuned are batch size and learning rate.

### Face detection and CNN classification model
This method is motivating because we believe that there is too much noise and not enough sample in the Twitter user profile picture dataset, so we decided to firstly train the CNN model on the UTK dataset, which is much larger and has less noise on the pictures.
##### Remove non-human profile pictures
Before we start training the CNN model, used RetinaFace(https://arxiv.org/abs/1905.00641)
to remove pictures that contains no human face. This will help reduce the noise when using the CNN model for predictions.
##### Model Description
The model used 3x3 Conv2D layers and 3x3 MaxPooling2D layers to extract image features. Then use a deeply connected Dense layer to produce outputs. We used ReLU as the activation function of the network, and softmax for the output layer. Each fold will perform 200 epochs of training. The initial learning rate is 1e-4. The optimizer is Adam.



