# -*- coding: utf-8 -*-
"""dl_a3_seq2seq

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/dl-a3-seq2seq-a0db6b52-0429-4bc8-a9f9-0a8139b80ad7.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240517/auto/storage/goog4_request%26X-Goog-Date%3D20240517T180005Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D79b21f650e0fe72390678ff90eb5108cc4dc853979591bfb564457a644cd31ea2f5a1a3a806e2bb1123a2cbd017d280f3af2233db282f74b279cbd457de2726777814f8406fa8d613cb8693fff584cee01d85792978f5a6ee328ff1d54bba22c3551e99826e1dd25b3b3affc06e27ae8ad3526f84ac18f6b96e9b5f74a94ce9d4c1a1ea24e83a4fad9287b64bb834d54791886477a64fd7fa8f04959c01fc9b556d0ef445337dd4453314aa080959192e48ff63cc8dea5c815d92480145941e5e4016a44b6d292ce88b3f019c20aa73d9ecaabecd4c448abeb341132652aeb629fd96ca0132d0515b9e074b2cc7718e9056fa2fc6df11282c1027f9fc7a67e9f
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

def load_data(path,batch_size = 32):
    df = pd.read_csv(path)
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

path1 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_train.csv'
custom_dataset1,train_loader_ben,a,b,_,_ = load_data(path1,batch_size = 64)
path2 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_valid.csv'
custom_dataset,val_loader_ben,_,_,_,_ = load_data(path2,batch_size = 64)
print(a,b)

"""# **Seq2Seq Bidirectional Model**"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Encoder class
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
                return torch.sum(hidden_states[:-2],dim= 0, keepdim = True), torch.sum(cell_states[-2:],dim= 0, keepdim = True)
            else:
                return hidden_states[-1].unsqueeze(0), cell_states[-1].unsqueeze(0)
        else:
            if self.bidirectional:
                return torch.sum(hidden[:-2],dim= 0, keepdim = True)
            else:
                return hidden[-1].unsqueeze(0)

# Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, decoder_layers=1, drop_prob=0.5, cell_type='gru'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.cell_type = cell_type
        self.dropout = nn.Dropout(drop_prob)
        self.embedding = nn.Embedding(output_size, embed_size)
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, decoder_layers, dropout=drop_prob, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output.squeeze(1), hidden

# Sequence to sequence class
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embed_size, encoder_layers=1, decoder_layers=1, drop_prob=0.3, cell_type='gru', bidirectional= True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, embed_size, encoder_layers, drop_prob, cell_type, bidirectional)
        self.decoder = Decoder(hidden_size, embed_size, output_size, decoder_layers, drop_prob, cell_type)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        output_vocab_size = output_size #self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, target_len, output_vocab_size).to(source.device)

        encoder_hidden = self.encoder(source)  # , encoder_cell
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)  # Initialize decoder hidden state
        decoder_input = target[:, 0]  # start token

        for t in range(1, target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            t1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else t1

        return outputs

    def _init_decoder_hidden(self, encoder_hidden):
        decoder_layers = self.decoder.decoder_layers
        encoder_layers = encoder_hidden[0].shape[0] if isinstance(encoder_hidden, tuple) else encoder_hidden.shape[0]

        if self.decoder.cell_type == 'lstm':
            if encoder_layers < decoder_layers:
                # Pad the encoder hidden state with zeros to match decoder_layers
                encoder_hidden = (
                    torch.cat(
                        [encoder_hidden[0], torch.zeros(decoder_layers - encoder_layers, *encoder_hidden[0].shape[1:], device=encoder_hidden[0].device)],
                        dim=0),torch.cat(
                        [encoder_hidden[1], torch.zeros(decoder_layers - encoder_layers, *encoder_hidden[1].shape[1:], device=encoder_hidden[1].device)],
                        dim=0))
            if encoder_hidden[0].shape[0] != decoder_layers:
                # If encoder layers and decoder layers are different, adjust the hidden state
                encoder_hidden = (encoder_hidden[0][:decoder_layers], encoder_hidden[1][:decoder_layers])

        else:
            if encoder_layers < decoder_layers:
                # Pad the encoder hidden state with zeros to match decoder_layers
                encoder_hidden = torch.cat(
                    [encoder_hidden, torch.zeros(decoder_layers - encoder_layers, *encoder_hidden.shape[1:], device=encoder_hidden.device)],
                    dim=0
                )
            if encoder_hidden.shape[0] != decoder_layers:
                # If encoder layers and decoder layers are different, adjust the hidden state
                encoder_hidden = encoder_hidden[:decoder_layers]

        return encoder_hidden

"""# **Train and evaluate**"""

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for latin, devanagari in dataloader:
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
    return model, total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for latin, devanagari in dataloader:  #tqdm(dataloader, desc='Evaluating', unit='batch'):
            latin = latin.to(device)
            devanagari = devanagari.to(device)

            output = model(latin, devanagari,teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]

            loss = criterion(output.view(-1, output_dim), devanagari.view(-1))
            total_loss += loss.item()

            max_values ,max_index = torch.max(output, 2) #output.argmax(dim=1)
            ind = max_index > 9
            max_index[ind] -= 2
#             print(f"prediction:{max_index} actual:{devanagari}")
            correct1=(max_index == devanagari).all(dim=1).sum().item()  # Calculate word accuracy
#
            total_correct += correct1
            total_samples += devanagari.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy*100

# Example usage
input_size = 30  # Number of Latin characters
output_size = 70  # Number of Devanagari characters
embed_size = 128
hidden_size = 128
encoder_layers = 3
decoder_layers = 2
cell_type = 'lstm'
batch_size = 64
num_epochs = 20
drop_prob = 0.2
learning_rate = 0.001
bidirectional = True


# Initialize the model, criterion, and optimizer
model = Seq2Seq(input_size, output_size, hidden_size,embed_size, encoder_layers,decoder_layers,drop_prob, cell_type,bidirectional)
print(model)

# model = Attention_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# for epoch in range(num_epochs):
#     trained_model, train_loss = train(model, train_loader_ben, criterion, optimizer, device)
#     val_loss, val_accuracy = evaluate(trained_model, val_loader_ben, criterion, device)
#     model = trained_model
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

"""# **Wandb Setup**"""

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
            'values': [1,2,3]
        },
        'decoder_layers':{
            'values': [1,2,3]
        },
        'hidden_layer_size':{
            'values': [16,32,64,128,256]
        },
        'cell_type': {
            'values': [ 'lstm', 'rnn', 'gru']
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

sweep_id = wandb.sweep(sweep=sweep_config, project='DL_assignment_3')

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
        path1 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_train.csv'
        custom_dataset1,train_loader_ben,a,b,_,_ = load_data(path1,batch_size = wandb.config.batch_size)
        path2 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_valid.csv'
        custom_dataset,val_loader_ben,_,_,_,_ = load_data(path2,batch_size = wandb.config.batch_size)

        # Training loop
        for epoch in range(wandb.config.num_epochs):
            trained_model, train_loss = train(model, train_loader_ben, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(trained_model, val_loader_ben, criterion, device)
            model = trained_model
            wandb.log({'Epoch': epoch, 'train_loss': train_loss , ' val_loss': val_loss, 'val_accuracy':val_accuracy})
            print(f'Epoch {epoch+1}/{wandb.config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


wandb.agent(sweep_id, function= main,count= 30) # calls main function for count number of times.
wandb.finish()

"""# **Best model**"""

# Best hyperparameter configuration
input_size = 30  # Number of Latin characters
output_size = 70  # Number of Devanagari characters
embed_size = 128
hidden_size = 256
encoder_layers = 3
decoder_layers = 3
cell_type = 'lstm'
batch_size = 64
num_epochs = 11
drop_prob = 0.2
learning_rate = 0.001
bidirectional = True

# Assuming you have loaded your dataset into train_loader and val_loader

# Initialize the model, criterion, and optimizer
Best_model = Seq2Seq(input_size, output_size, hidden_size,embed_size, encoder_layers,decoder_layers,drop_prob, cell_type,bidirectional)
print(Best_model)

# model = Attention_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Best_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Best_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    trained_model, train_loss = train(Best_model, train_loader_ben, criterion, optimizer, device)
    model = trained_model

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

path3 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_test.csv'
c, test_loader_ben, input_vocab, target_vocab, max_length, _ = load_data(path3, batch_size=64)  # Use correct path3

val_loss, val_accuracy = evaluate(trained_model, test_loader_ben, criterion, device)
print(f' Test Accuracy: {val_accuracy:.4f}')

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

path3 = '/kaggle/input/aksharantar/aksharantar_sampled/ben/ben_test.csv'
c, test_loader_ben, input_vocab, target_vocab, max_length, _ = load_data(path3, batch_size=64)  # Use correct path3

# Make sure to define the reverse dictionaries for converting indices back to text
latin_idx2token = {idx: char for char, idx in input_vocab.items()}
bangla_idx2token = {idx: char for char, idx in target_vocab.items()}

import csv
with open('seq2seq_results.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Input Text', 'Actual Target', 'Predicted Text'])
    writer.writerows(seq2seq_results)

df1 = pd.read_csv('seq2seq_results.csv')
df1

test_predictions, actual = pred(model, test_loader_ben, device)
seq2seq_results = []
for (src_indices, output_indices),act_ind in zip(test_predictions,actual):
#     print(src_indices)
#     print('\njkl',output_indices)
    # Since our data loader might have batch size greater than 1, iterate through each example in the batch
    for i in range(src_indices.shape[0]):
        input_text = decode_indices(src_indices[i], latin_idx2token, input_vocab)
        actual_target_text = decode_indices(act_ind[i], bangla_idx2token, target_vocab)
        predicted_text = decode_indices_target(output_indices[i], bangla_idx2token, target_vocab)
        seq2seq_results.append([input_text, actual_target_text, predicted_text])
        print(f'SL. {i} Input Text: {input_text} -> Actual target: {actual_target_text} -> Predicted Text: {predicted_text}')




