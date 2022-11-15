import argparse
import datetime
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class RNN_Model(nn.Module):
    def __init__(self, num_classes, max_features, embedding_dim, hidden_size=128, device='cpu'):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = 3

        self.embedding = nn.Embedding(max_features, self.embedding_dim)

        self.dropout = nn.Dropout(0.35)

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=0.6)

        self.output = nn.Linear(self.hidden_size*2, num_classes)

    def forward(self, word_seq, h_init):
        embedded = self.embedding(word_seq)
        embedded = torch.transpose(embedded, 0, 1)

        h_seq, h_final = self.rnn(embedded, h_init)
        score_seq = torch.tanh(self.dropout(self.output(h_seq)))

        return score_seq[-1], h_final

def train(model, train_X, train_Y, test_X, test_Y, tokenizer, encoder, batch_size, learning_rate, num_epochs, device='cpu', model_path=None):

    train_X = torch.tensor(train_X, dtype=torch.long).to(device)
    train_Y = torch.tensor(train_Y, dtype=torch.long).to(device)
    test_X = torch.tensor(test_X, dtype=torch.long).to(device)
    test_Y = torch.tensor(test_Y, dtype=torch.long).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # assign these variables
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    h = torch.zeros(model.num_layers*2, batch_size, model.hidden_size).to(device)
    c = torch.zeros(model.num_layers*2, batch_size, model.hidden_size).to(device)
    final_acc = []

    start = datetime.datetime.now()
    print("Training started at: ", start)

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        true_count = 0
        total_count = 0
        accuracies = []
        for step, data in enumerate(train_loader):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            if texts.shape[0] != batch_size:
                continue

            # zero the parameter gradients
            optimizer.zero_grad()
            # do forward propagation
            h = h.detach()
            c = c.detach()
            probabilities, hc = model(texts, (h, c))
            h, c = hc
            # do loss calculation
            loss = criterion(probabilities, labels)
            # do backward propagation
            loss.backward()
            # do parameter optimization step
            normalize_gradient(model)
            optimizer.step()

            # Calculate the count of accurate predictions
            true_count += torch.sum(torch.argmax(probabilities, dim=1) == labels)
            total_count += labels.shape[0]

            # calculate running loss value for non padding
            running_loss += loss.item()
            accuracy = (true_count/total_count)*100
            accuracies.append(accuracy)
            if (step+1)%50 == 0:
                print('epoch: {}, step: {}, loss: {}, accuracy: {}'.format(epoch+1, step+1, running_loss/(step+1),accuracy))
        scheduler.step()
        final_acc.append((sum(accuracies)*1.0/len(accuracies)))
        
        # Turn off model training for validation
        model.eval()
        running_vloss = 0.0
        true_count = 0
        total_count = 0
        val_accuracies = []
        h_val = torch.zeros(model.num_layers*2, batch_size, model.hidden_size).to(device)
        c_val = torch.zeros(model.num_layers*2, batch_size, model.hidden_size).to(device)
        for step, vdata in enumerate(validation_loader):
            
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            if vinputs.shape[0] != batch_size:
                continue

            # Use the current model to get outputs for the validation inputs
            h_val = h_val.detach()
            c_val = c_val.detach()
            voutputs, hc_val = model(vinputs, (h_val, c_val))
            h_val, c_val = hc_val

            # Get the validation loss
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

            # Calculate the count of accurate predictions
            true_count += torch.sum(torch.argmax(voutputs, dim=1) == vlabels)
            total_count += vlabels.shape[0]

        # Calculate average loss over all steps
        avg_vloss = running_vloss / (step + 1)
        accuracy = (true_count/total_count)*100
        val_accuracies.append(accuracy)
        print(f'validation, epoch: {epoch+1}, loss: {avg_vloss}, accuracy: {accuracy}')

    end = datetime.datetime.now()
    print("Training ended at: ", end)
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'vocabulary': {
            'texts': list(tokenizer.word_index.keys()),
            'labels': list(encoder.classes_)
        }
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

def normalize_gradient(model):

    norm_sq=0

    for param in model.parameters():
        norm_sq += param.grad.data.norm() ** 2
    grad_norm=torch.sqrt(norm_sq)

    if grad_norm<1e-4:
        model.zero_grad()
    else:    
        for param in model.parameters():
            param.grad.data.div_(grad_norm)

    return grad_norm


def process_texts(train_X, test_X, max_features, max_len):
    # Create tokenizer
    tokenizer = Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(list(train_X))

    # Tokenize the words in the training and test summaries
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the tokenized arrays 
    train_X = pad_sequences(train_X, maxlen = max_len)
    test_X = pad_sequences(test_X, maxlen = max_len)

    return train_X, test_X, tokenizer

def process_labels(train_Y, test_Y):
    # Create label encoder
    encoder = LabelEncoder()

    # Encode the training and test labels
    train_Y = encoder.fit_transform(train_Y.values)
    test_Y = encoder.transform(test_Y.values)

    return train_Y, test_Y, encoder



def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    movies = pd.read_csv(args.data_path, index_col=0)
    data = movies[['summary', 'labelled_genre']]
    train_X, test_X, train_Y, test_Y = train_test_split(data['summary'], data['labelled_genre'], 
                                                        stratify=data['labelled_genre'], test_size = 0.3)

    
    learning_rate = 1e-3
    batch_size = 600
    num_epochs = 20
    max_features = 12000
    max_len = 50
    embedding_dim = 1024

    train_X, test_X, tokenizer = process_texts(train_X, test_X, max_features, max_len)
    train_Y, test_Y, encoder = process_labels(train_Y, test_Y)
    num_classes = len(encoder.classes_)
    model = RNN_Model(num_classes, max_features, embedding_dim, 96, device_str).to(device)
    model.embedding.weight.data.uniform_(-0.1, 0.1)
    model.output.weight.data.uniform_(-0.1, 0.1)

    train(model, train_X, train_Y, test_X, test_Y, tokenizer, encoder, batch_size, learning_rate, num_epochs, device, args.model_path)

    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to the data file')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
