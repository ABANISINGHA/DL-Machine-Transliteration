# -*- coding: utf-8 -*-
"""dl_A3_Attention

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/dl-a3-attention-ff17ae91-dcf1-4de0-960c-eec618c439c1.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240517/auto/storage/goog4_request%26X-Goog-Date%3D20240517T165117Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4c4e932f9842173679419a54191e0ca0a69f6bf260fccd2040351f47e6a941c03e2d1301fca389efd6775560fb61bc0e76f68b019a94223db3cae2a07ef460225505cb4ea29172336c1131e5f02489e1a09b18c3dfe96c2396b9250f0a41774bda77d1ed59b5f47e11d04da55b9a6c362bf661a7c7cb2b8bfef7b1965a3dc61db306f6482952530293706a9acf00449403fea53aebc0b8da6076b8cdc31337dd2c4851eee607ad98bcd18ad6150112526122814399f478b4b83d14e5231daf18f1036f21623decd7eb4f0ea6f71d40cf86fe4a4ce781097d8a8ab1670d296745165a73b473a493cf4d2bf54dcb82c65517ed37e337a263d37b48ab368eea5a8a
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Function for dataload
def load_data(path,batch_size = 32):
    df = pd.read_csv(path)
#     df = df.head(10)
    df.columns = ['input_word','target_word']

    # Define maximum sequence lengths for letters
    max_input_len = max(len(word) for word in df['input_word'])
    max_target_len = max(len(word) for word in df['target_word'])

    # Define vocabulary mappings for letters
    input_letter_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}  # Add special tokens
    target_letter_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}  # Add special tokens
    letter_idx = 3

    input_str = ''
    target_str = ''
    # Preprocess the data and update vocabulary mappings for letters
    for input_word, target_word in zip(df['input_word'], df['target_word']):
        input_str += input_word
        target_str += target_word


    # Update vocabulary mappings for input letters
    for letter in sorted(set(input_str)):
        input_letter_vocab[letter] = letter_idx
        letter_idx += 1
    letter_idx = 3
    # Update vocabulary mappings for target letters
    for letter in sorted(set(target_str)):
        if letter not in target_letter_vocab:
            target_letter_vocab[letter] = letter_idx
            letter_idx += 1

    # Tokenize function at the letter level
    def tokenize_input_letters(word, vocab, max_len):
        token_ids = [vocab[char] for char in word if char in vocab]
        padded = token_ids[:max_len] + [vocab['<pad>']] * (max_len - len(token_ids))
        return torch.tensor(padded)

    def tokenize_target_letters(word, vocab, max_len):
        token_ids = [vocab[char] for char in word if char in vocab]
        padded =  [vocab['<pad>']]+ token_ids[:max_len] +[vocab['<pad>']] * (max_len - len(token_ids))
        return torch.tensor(padded)

#  [vocab['<sos>']]+[vocab['<eos>']]
    # Custom Dataset class for letter-level tokenization
    class CustomDataset(Dataset):
        def __init__(self, input_data, target_data, input_vocab, target_vocab, max_input_len, max_target_len):
            self.input_data = input_data
            self.target_data = target_data
            self.input_vocab = input_vocab
            self.target_vocab = target_vocab
            self.max_input_len = max_input_len
            self.max_target_len = max_target_len

        def __len__(self):
            return len(self.input_data)

        def __getitem__(self, idx):
            input_word = self.input_data[idx]
            target_word = self.target_data[idx]

            # Tokenize input and target words at the letter level
            input_letters = tokenize_input_letters(input_word, self.input_vocab, self.max_input_len)
            target_letters = tokenize_target_letters(target_word, self.target_vocab, self.max_target_len)

            return input_letters, target_letters

    # Create DataLoader
    custom_dataset = CustomDataset(df['input_word'], df['target_word'], input_letter_vocab, target_letter_vocab, max_input_len, max_target_len)
    data_loader1 = DataLoader(custom_dataset, batch_size=batch_size, shuffle = False )

    return custom_dataset,data_loader1, input_letter_vocab, target_letter_vocab, max_input_len, max_target_len

path1 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_train.csv'
custom_dataset1,train_loader_ben,a,b,_,_ = load_data(path1,batch_size = 64)
path2 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_valid.csv'
custom_dataset,val_loader_ben,_,_,_,_ = load_data(path2,batch_size = 64)
print(a,b)

"""# **Attention Model**"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, encoder_layers=1, drop_prob=0.5, cell_type='gru', bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(drop_prob)
        self.embedding = nn.Embedding(input_size, embed_size)

        if cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, encoder_layers, dropout=drop_prob, bidirectional=bidirectional, batch_first=True)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, encoder_layers, dropout=drop_prob, bidirectional=bidirectional, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, encoder_layers, dropout=drop_prob, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded)

        if self.cell_type == 'lstm':
            hidden_states, cell_states = hidden
            if self.bidirectional:
                hidden = (torch.cat([hidden_states[-2], hidden_states[-1]], dim=1).unsqueeze(0),
                          torch.cat([cell_states[-2], cell_states[-1]], dim=1).unsqueeze(0))
            else:
                hidden = (hidden_states[-1].unsqueeze(0), cell_states[-1].unsqueeze(0))
        else:
            if self.bidirectional:
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(0)
            else:
                hidden = hidden[-1].unsqueeze(0)

        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)

        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)

        attention_weights = torch.bmm(v, energy)
        return torch.softmax(attention_weights.squeeze(1), dim=1)

class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, attention, decoder_layers=1, drop_prob=0.5, cell_type='gru'):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.cell_type = cell_type
        self.attention = attention
        self.dropout = nn.Dropout(drop_prob)
        self.embedding = nn.Embedding(output_size, embed_size)

        input_size = hidden_size + embed_size
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        attention_weights = self.attention(hidden[-1], encoder_outputs)
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)

        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
        else:
            output, hidden = self.rnn(rnn_input, hidden)

        output = self.fc(torch.cat((output, context), dim=2).squeeze(1))

        if self.cell_type == 'lstm':
            return output, (hidden, cell)
        else:
            return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embed_size, encoder_layers=1, decoder_layers=1, drop_prob=0.3, cell_type='gru', bidirectional=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, embed_size, encoder_layers, drop_prob, cell_type, bidirectional)
        self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)
        self.decoder = DecoderWithAttention(hidden_size * 2 if bidirectional else hidden_size, embed_size, output_size, self.attention, decoder_layers, drop_prob, cell_type)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        output_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, target_len, output_vocab_size).to(source.device)

        encoder_outputs, encoder_hidden = self.encoder(source)
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        decoder_input = target[:, 0]

        for t in range(1, target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[:, t] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            t1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else t1

        return outputs

    def _init_decoder_hidden(self, encoder_hidden):
        decoder_layers = self.decoder.decoder_layers
        if self.encoder.cell_type == 'lstm':
            encoder_hidden = (
                torch.cat([encoder_hidden[0][i] for i in range(encoder_hidden[0].shape[0])], dim=1).unsqueeze(0),
                torch.cat([encoder_hidden[1][i] for i in range(encoder_hidden[1].shape[0])], dim=1).unsqueeze(0)
            )
            if encoder_hidden[0].shape[0] != decoder_layers:
                encoder_hidden = (
                    encoder_hidden[0][:decoder_layers],
                    encoder_hidden[1][:decoder_layers]
                )
        else:
            encoder_hidden = torch.cat([encoder_hidden[i] for i in range(encoder_hidden.shape[0])], dim=1).unsqueeze(0)
            if encoder_hidden.shape[0] != decoder_layers:
                encoder_hidden = encoder_hidden[:decoder_layers]

        return encoder_hidden

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for latin, devanagari in tqdm(dataloader, desc='Training', unit='batch'):
        latin = latin.to(device)
        devanagari = devanagari.to(device)

        optimizer.zero_grad()

        output = model(latin, devanagari)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        devanagari = devanagari.view(-1)

        loss = criterion(output, devanagari)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Reshape the output and target to their original shape
        output = output.view(latin.size(0), -1, output_dim)
        devanagari = devanagari.view(latin.size(0), -1)

        max_index = output.argmax(dim=2)
        # Calculate word-level accuracy
        correct = (max_index == devanagari).all(dim=1).sum().item()
        total_correct += correct
        total_samples += devanagari.size(0)

    accuracy = total_correct / total_samples
    return model, total_loss / len(dataloader), accuracy*100


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for latin, devanagari in tqdm(dataloader, desc='Evaluating', unit='batch'):
            latin = latin.to(device)
            devanagari = devanagari.to(device)

            output = model(latin, devanagari, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            devanagari = devanagari.view(-1)

            loss = criterion(output, devanagari)
            total_loss += loss.item()

            # Reshape the output and target to their original shape
            output = output.view(latin.size(0), -1, output_dim)
            devanagari = devanagari.view(latin.size(0), -1)

            max_index = output.argmax(dim=2)
            mask = max_index > 9
            max_index[mask] -= 3
            # Calculate word-level accuracy
            correct = (max_index == devanagari).all(dim=1).sum().item()
            total_correct += correct
            total_samples += devanagari.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy * 100

# Example usage
input_size = 30  # Number of Latin characters
output_size = 70  # Number of Devanagari characters
embed_size = 16
hidden_size = 32
encoder_layers = 1
decoder_layers = 1
cell_type = 'rnn'
batch_size = 64
num_epochs = 12
drop_prob = 0.3
learning_rate = 0.001

# Assuming you have loaded your dataset into train_loader and val_loader

# Initialize the model, criterion, and optimizer
model = Seq2Seq(input_size, output_size, hidden_size,embed_size, encoder_layers,decoder_layers,drop_prob, cell_type)
print(model)

# model = Attention_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
ignore_index = 0
criterion = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# for epoch in range(num_epochs):
#     trained_model, train_loss, train_acc  = train(model, train_loader_ben, criterion, optimizer, device)
#     val_loss, val_accuracy = evaluate(trained_model, val_loader_ben, criterion, device)
#     model = trained_model
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train_acc: {train_acc},  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# !pip install wandb
import wandb
import numpy as np
from types import SimpleNamespace
import random

wandb.login(key='bb3c7761be2856a8335d16d1483149380482ae9e')#bb3c7761be2856a8335d16d1483149380482ae9e

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'embedding_size':{
            'values': [16,32,64,128,256]
        },
        'dropout': {
            'values': [0.3, 0.2,0.5]
        },
        'encoder_layers': {
            'values': [1]
        },
        'decoder_layers':{
            'values': [1]
        },
        'hidden_layer_size':{
            'values': [16,32,64,128,256]
        },
        'cell_type': {
            'values': [ 'lstm','rnn', 'gru']
        },
        'bidirectional': {
            'values': [True, False]
        },
        'batch_size': {
            'values': [32,64]
        },
        'num_epochs': {
            'values': [10,12]
        },
        'learning_rate': {
            'values': [0.01,0.001]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DL_A3_Attention')

def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''

    with wandb.init() as run:
        run_name="ct-"+str(wandb.config.cell_type)+"_el-"+str(wandb.config.encoder_layers)+"_dl-"+str(wandb.config.decoder_layers)+"_drop-"+str(wandb.config.dropout)+"_es-"+str(wandb.config.embedding_size)+"_hs-"+str(wandb.config.hidden_layer_size)+"_bs-"+str(wandb.config.batch_size)+"_ep-"+str(wandb.config.num_epochs)+"lr"+str(wandb.config.learning_rate)
        wandb.run.name=run_name


        model = Seq2Seq(input_size=30, output_size=70, hidden_size=wandb.config.hidden_layer_size,embed_size=wandb.config.embedding_size,encoder_layers=wandb.config.encoder_layers,
                        decoder_layers=wandb.config.decoder_layers,drop_prob=wandb.config.dropout, cell_type=wandb.config.cell_type, bidirectional=wandb.config.bidirectional)
        print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        path1 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_train.csv'
        custom_dataset1,train_loader_ben,a,b,_,_ = load_data(path1,batch_size = wandb.config.batch_size)
        path2 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_valid.csv'
        custom_dataset,val_loader_ben,_,_,_,_ = load_data(path2,batch_size = wandb.config.batch_size)

        # Training loop
        for epoch in range(wandb.config.num_epochs):
            trained_model, train_loss, train_acc = train(model, train_loader_ben, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(trained_model, val_loader_ben, criterion, device)
            model = trained_model
            wandb.log({'Epoch': epoch, 'train_loss': train_loss , ' val_loss': val_loss, 'val_accuracy':val_accuracy})
            print(f'Epoch {epoch+1}/{wandb.config.num_epochs}, Train Loss: {train_loss:.4f},Train_acc: {train_acc} Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


#         model_train(model,train,validation)

wandb.agent(sweep_id, function= main,count= 30) # calls main function for count number of times.
wandb.finish()

"""# **Best model**"""

# Best configutration
input_size = 30  # Number of Latin characters
output_size = 70  # Number of Devanagari characters
embed_size = 256
hidden_size = 256
encoder_layers = 1
decoder_layers = 1
cell_type = 'lstm'
batch_size = 64
num_epochs = 9
drop_prob = 0.3
learning_rate = 0.001
bidirectional=False

# Assuming you have loaded your dataset into train_loader and val_loader

# Initialize the model, criterion, and optimizer
Best_model = Seq2Seq(input_size, output_size, hidden_size,embed_size, encoder_layers,decoder_layers,drop_prob, cell_type, bidirectional)
print(Best_model)

# model = Attention_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Best_model.to(device)
ignore_index = 0
criterion = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(Best_model.parameters(), lr=learning_rate)

path3 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_test.csv'
c, test_loader_ben, input_vocab, target_vocab, max_length, _ = load_data(path3, batch_size=64)  # Use correct path3


# Training loop
for epoch in range(num_epochs):
    trained_model, train_loss,_ = train(Best_model, train_loader_ben, criterion, optimizer, device)
    model = trained_model

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

val_loss, val_accuracy = evaluate(trained_model, test_loader_ben, criterion, device)
print(f' Test Accuracy: {val_accuracy:.4f}')

"""# **Prediction on test dataset for best model**"""

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'embedding_size':{
            'values': [256]
        },
        'dropout': {
            'values': [0.3]
        },
        'encoder_layers': {
            'values': [1]
        },
        'decoder_layers':{
            'values': [1]
        },
        'hidden_layer_size':{
            'values': [256]
        },
        'cell_type': {
            'values': ['lstm']
        },
        'bidirectional': {
            'values': [True]
        },
        'batch_size': {
            'values': [64]
        },
        'num_epochs': {
            'values': [9]
        },
        'learning_rate': {
            'values': [0.001]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DL_A3_Attention')

def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''

    with wandb.init() as run:
        run_name="ct-"+str(wandb.config.cell_type)+"_el-"+str(wandb.config.encoder_layers)+"_dl-"+str(wandb.config.decoder_layers)+"_drop-"+str(wandb.config.dropout)+"_es-"+str(wandb.config.embedding_size)+"_hs-"+str(wandb.config.hidden_layer_size)+"_bs-"+str(wandb.config.batch_size)+"_ep-"+str(wandb.config.num_epochs)+"lr"+str(wandb.config.learning_rate)
        wandb.run.name=run_name


        model = Seq2Seq(input_size=30, output_size=70, hidden_size=wandb.config.hidden_layer_size,embed_size=wandb.config.embedding_size,encoder_layers=wandb.config.encoder_layers,
                        decoder_layers=wandb.config.decoder_layers,drop_prob=wandb.config.dropout, cell_type=wandb.config.cell_type, bidirectional=wandb.config.bidirectional)
        print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        path1 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_train.csv'
        custom_dataset1,train_loader_ben,a,b,_,_ = load_data(path1,batch_size = wandb.config.batch_size)
        path2 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_valid.csv'
        custom_dataset,val_loader_ben,_,_,_,_ = load_data(path2,batch_size = wandb.config.batch_size)
        path3 = '/kaggle/input/aksharantar-sampled-dataset/aksharantar_sampled/ben/ben_test.csv'
        c, test_loader_ben, input_vocab, target_vocab, max_length, _ = load_data(path3, batch_size=64)  # Use correct path3

        # Training loop
        for epoch in range(wandb.config.num_epochs):
            trained_model, train_loss, train_acc = train(model, train_loader_ben, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(trained_model, test_loader_ben, criterion, device)
            model = trained_model
#             wandb.log({'Epoch': epoch, 'train_loss': train_loss , ' val_loss': val_loss, 'val_accuracy':val_accuracy})
            print(f'Epoch {epoch+1}/{wandb.config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


#         model_train(model,train,validation)

wandb.agent(sweep_id, function= main,count= 1) # calls main function for count number of times.
wandb.finish()

"""# **Prediction**"""

def decode_indices(indices, idx2token, target_vocab):
    valid_indices = []
    for idx in indices:
        if idx in idx2token and idx not in (target_vocab['<pad>'], target_vocab['<sos>'], target_vocab['<eos>']):
            valid_indices.append(idx)
#             print(valid_indices)
    decoded_text = ''
    for idx in valid_indices:
        decoded_text += idx2token[idx]
#         print(decoded_text)
    return decoded_text

def decode_indices_target(indices, idx2token, target_vocab):
    valid_indices = []
    for idx in indices:
        if idx in idx2token and idx not in (target_vocab['<pad>'], target_vocab['<sos>'], target_vocab['<eos>']):
            if idx < 10:
                valid_indices.append(idx)
            else:
                valid_indices.append(idx-3)
#             print(valid_indices)
    decoded_text = ''
    for idx in valid_indices:
        decoded_text += idx2token[idx]
#         print(decoded_text)
    return decoded_text

def decode_indices_target1(indices1,indices2, idx2token, target_vocab):
    valid_indices1 = []
    for idx in indices1:
        if idx in idx2token and idx not in (target_vocab['<pad>'], target_vocab['<sos>'], target_vocab['<eos>']):
            valid_indices1.append(idx)
#             print(valid_indices)
    valid_indices2 = []
    for idx in indices2:
        if idx in idx2token and idx not in (target_vocab['<pad>'], target_vocab['<sos>'], target_vocab['<eos>']):
            if idx < 10:
                valid_indices2.append(idx)
            else:
                valid_indices2.append(idx-3)
#             print(valid_indices)
    decoded_text1 = ''
    decoded_text2 = ''
    l1 = len(valid_indices1)
    val_ind2 = valid_indices2[:l1]
    for idx2 in val_ind2:
#         decoded_text1 += idx2token[idx1]
        decoded_text2 += idx2token[idx2]

#     decoded_text2 = ''
#     for idx in valid_indices:
#         decoded_text += idx2token[idx]
# #         print(decoded_text)
    return decoded_text2

def pred(model, dataloader, device):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for latin, devanagari in dataloader:#, desc='Evaluating', unit='batch'):
            latin = latin.to(device)
            devanagari = devanagari.to(device)
            output = model(latin, devanagari,0)
            deb = devanagari.cpu().numpy()
            actual.append(deb)
            output = output.argmax(2)
            latin = latin.cpu().numpy()
            output = output.cpu().numpy()
            predictions.append((latin, output))
    return predictions, actual


# Make sure to define the reverse dictionaries for converting indices back to text
latin_idx2token = {idx: char for char, idx in input_vocab.items()}
bangla_idx2token = {idx: char for char, idx in target_vocab.items()}

test_predictions, actual = pred(trained_model, test_loader_ben, device)
results = []
for (src_indices, output_indices),act_ind in zip(test_predictions,actual):
#     print(src_indices)
#     print('\njkl',output_indices)
    # Since our data loader might have batch size greater than 1, iterate through each example in the batch
    for i in range(src_indices.shape[0]):
        input_text = decode_indices(src_indices[i], latin_idx2token, input_vocab)
        actual_target_text = decode_indices(act_ind[i], bangla_idx2token, target_vocab)
        predicted_text = decode_indices_target1(act_ind[i],output_indices[i], bangla_idx2token, target_vocab)#decode_indices_target(output_indices[i], bangla_idx2token, target_vocab)
        results.append([input_text, actual_target_text, predicted_text])

        print(f'SL. {i} Input Text: {input_text} -> Actual target: {actual_target_text} -> Predicted Text: {predicted_text}')
#     break

# Writing results to CSV
import csv
with open('results.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Input Text', 'Actual Target', 'Predicted Text'])
    writer.writerows(results)

df1 = pd.read_csv('results.csv')
df1

