import argparse
import datetime
import math
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize

torch.manual_seed(0)
MAX_TOKEN_LENGTH = 50
MOST_COMMON = 1000
EMBEDDING_SIZE = 100


class LangDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None):
        self.texts = texts
        self.labels = labels

        # Training phase
        if vocab is None:
            self.text_vocab = self.create_text_vocab(self.texts)
            self.label_vocab = self.create_label_vocab(self.labels)
        else:
            # Testing phase
            self.text_vocab = vocab['texts']
            self.label_vocab = vocab['labels']


    def create_text_vocab(self, data):
        # Vocabulary starts with 1, 0 is for padding
        data = [word_tokenize(x) for x in data]

        text_vocab = {}
        fdist = nltk.FreqDist() 
        for sentence in data:
            for word in sentence:
                fdist[word] += 1
        
        common_words = fdist.most_common(MOST_COMMON)
        for idx, word in enumerate(common_words):
            text_vocab[word[0]] = (idx+1)

        return text_vocab


    def create_label_vocab(self, data):
        label_vocab = {}
        for label in data:
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
        
        return label_vocab


    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.text_vocab)
        num_class = len(self.label_vocab)

        return num_vocab, num_class
    

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)


    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        words = self.texts[i]
        text = []
        for word in words:
            if word in self.text_vocab.keys():
                text.append(self.text_vocab[word])

        if self.labels is None:
            return text

        label = self.label_vocab[self.labels[i]]
        return text, label


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    data = []
    labels = []

    for items in batch:
        # Testing phase only contains texts
        if type(items) == list:
            data.append(torch.tensor(items))
        else:
            # Training phase contains both texts and labels
            data.append(torch.tensor(items[0]))
            labels.append(items[1])

    # Pad all tensors to match length of the longest tensor 
    n = MAX_TOKEN_LENGTH - len(data[0])
    if n > 0:
        data[0] = torch.concat((data[0], torch.zeros(n)), 0)
    padded = pad_sequence(data, batch_first=True)
    if padded.shape[1] > MAX_TOKEN_LENGTH:
        padded = padded[:, :MAX_TOKEN_LENGTH]
    texts = torch.tensor(padded).long()

    if len(labels) != 0:
        labels = torch.LongTensor(labels)
    else:
        labels = None
    
    return texts, labels


class Model(nn.Module):
    def __init__(self, num_vocab, num_class, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab+1, EMBEDDING_SIZE, padding_idx=0)

        self.input = nn.Linear(EMBEDDING_SIZE*MAX_TOKEN_LENGTH, 50)
        self.h1 = nn.Linear(50, 50)
        self.h2 = nn.Linear(50, 20)
        self.output = nn.Linear(20, num_class)

        self.flatten = nn.Flatten(-2, -1)

        # Regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = self.flatten(embedded)
        
        x = F.relu(self.input(h0))
        x = F.relu(self.h1(x))
        x = F.relu(self.dropout(self.h2(x)))

        output = self.output(x)
        return output


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None, train_validate_split=0.8):
    train_size = math.floor(train_validate_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)

    # assign these variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    print("Training started at: ", start)

    for epoch in range(num_epoch):
        model.train(True)
        running_loss = 0.0
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

            # calculate running loss value for non padding
            running_loss += loss.item()
            if ((step+1) % 100) == 0:
                 print('epoch: {}, step: {}, loss: {}'.format(epoch+1, step+1, running_loss/(step+1)))
            #print('epoch: {}, step: {}, loss: {}'.format(epoch+1, step+1, running_loss/(step+1)))
        
        # Turn off model training for validation
        model.train(False)
        running_vloss = 0.0
        for step, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            # Use the current model to get outputs for the validation inputs
            voutputs = model(vinputs)

            # Get the validation loss
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

        # Calculate average loss over all steps
        avg_vloss = running_vloss / (step + 1)
        print(f'validation, epoch: {epoch+1}, loss: {avg_vloss}')

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
            'texts': dataset.text_vocab,
            'labels': dataset.label_vocab
        }
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=50, collate_fn=collator, shuffle=False)

    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts)

            # get the label predictions
            predictions = torch.argmax(outputs, dim=1).tolist()
            for p in predictions:
                labels.append(class_map[p])

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"

    if args.train:
        movies = pd.read_csv(args.data_path, index_col=0)
        summaries = list(movies['summary'])
        labels = list(movies['labelled_genre'])

        dataset = LangDataset(summaries, labels)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        
        learning_rate = 5e-4
        batch_size = 100
        num_epochs = 30

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)

    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # create the test dataset object using LangDataset class
        movies = pd.read_csv(args.data_path, index_col=0)
        summaries = list(movies['summary'])
        labels = list(movies['labelled_genre'])

        checkpoint = torch.load(args.model_path)
        dataset = LangDataset(summaries, vocab=checkpoint['vocabulary'])
        num_vocab, num_class = dataset.vocab_size()

        # initialize and load the model
        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        label_vocab = checkpoint['vocabulary']['labels']
        label_map = {v: k for k, v in label_vocab.items()}

        # run the prediction
        preds = test(model, dataset, label_map, device)
        
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))

        assert len(preds) == len(labels), \
            "Length of predictions ({}) and labels ({}) are not the same".format(len(preds), len(labels))
        correct = 0
        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
        print('Accuracy: {:.2f}'.format(float(correct) * 100 / len(labels)))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to the data file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)

# Accuracy: 17.10