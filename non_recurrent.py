import argparse
import datetime
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

class CNN_Model(nn.Module):
    
    def __init__(self, num_classes, max_features, embedding_dim, device='cpu'):
        super(CNN_Model, self).__init__()
        kernel_sizes = [1,2,3,5]
        num_filters = 36
        n_classes = num_classes
        self.embedding = nn.Embedding(max_features, embedding_dim)
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embedding_dim)).to(device) for K in kernel_sizes])
        # self.max_pools = nn.ModuleList([nn.MaxPool1d(i, i.size(2)).squeeze(2) for i in x])
        self.dropout = nn.Dropout(0.7)
        self.linear = nn.Linear(len(kernel_sizes)*num_filters, n_classes)


    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        outputs = self.linear(x) 
        return outputs

class MLP_Model(nn.Module):

    def __init__(self, num_classes, max_features, embedding_dim, max_len, device='cpu'):
        super(MLP_Model, self).__init__()
        neuron_sizes = [(max_len*embedding_dim, 1000), (1000, 500), (500, 250), (250, num_classes)]

        self.embedding = nn.Embedding(max_features, embedding_dim)
        self.linears = nn.ModuleList([nn.Linear(in_f, out_f).to(device) for (in_f, out_f) in neuron_sizes])
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten(-2, -1)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.flatten(x)
        for layer in self.linears:
            if x.shape[1] < 1000:
                x = torch.tanh(self.dropout(layer(x)))
            else:
                x = torch.tanh(layer(x))
        outputs = x

        return outputs


def train(model, train_X, train_Y, test_X, test_Y, tokenizer, encoder, batch_size, learning_rate, num_epochs, device='cpu', model_path=None):

    train_X = torch.tensor(train_X, dtype=torch.long).to(device)
    train_Y = torch.tensor(train_Y, dtype=torch.long).to(device)
    test_X = torch.tensor(test_X, dtype=torch.long).to(device)
    test_Y = torch.tensor(test_Y, dtype=torch.long).to(device)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # assign these variables
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    final_acc = []

    start = datetime.datetime.now()
    print("Training started at: ", start)

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        true_count, total_count, accuracies = 0, 0, []
        for step, data in enumerate(train_loader):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # do forward propagation
            probabilities = model(texts)
            # do loss calculation
            loss = criterion(probabilities, labels)
            # do backward propagation
            loss.backward()
            # do parameter optimization step
            optimizer.step()

             # Calculate the count of accurate predictions
            true_count += torch.sum(torch.argmax(probabilities, dim=1) == labels)
            total_count += labels.shape[0]

            # calculate running loss value for non padding
            running_loss += loss.item()
            accuracy = (true_count/total_count)*100
            accuracies.append(accuracy)
            if (step+1)%100 == 0:
                print('epoch: {}, step: {}, loss: {},accuracy: {}'.format(epoch+1, step+1, running_loss/(step+1), accuracy))
        final_acc.append(sum(accuracies)*1.0/len(accuracies))
        
        # Turn off model training for validation
        model.eval()
        running_vloss = 0.0
        true_count = 0
        total_count = 0
        val_accuracies = []
        for step, vdata in enumerate(validation_loader):
            
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            # Use the current model to get outputs for the validation inputs
            voutputs = model(vinputs)

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
        print(f'validation, epoch: {epoch+1}, loss: {avg_vloss}, accuracy: {(true_count/total_count)*100}')

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
    encoder = LabelEncoder()
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
    batch_size = 200
    num_epochs = 20
    max_features = 7000
    max_len = 100
    embedding_dim = 32

    train_X, test_X, tokenizer = process_texts(train_X, test_X, max_features, max_len)
    train_Y, test_Y, encoder = process_labels(train_Y, test_Y)
    num_classes = len(encoder.classes_)
    model = CNN_Model(num_classes, max_features, embedding_dim, device_str).to(device)
    # model = MLP_Model(num_classes, max_features, embedding_dim, max_len, device_str).to(device)

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