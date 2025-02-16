# DL_Assignment_3
# MA23M001 Abani Singha

# Seq2Seq Transliteration Model

This repository contains the code and resources for a sequence-to-sequence (seq2seq) transliteration model developed using the Aksharantar dataset on Kaggle. The primary goal of this project is to transliterate text from one script to another using a neural network model.

## Overview

Transliteration is the process of converting text from one script to another while preserving its phonetic characteristics. This project utilizes a seq2seq model with attention mechanism to achieve effective and accurate transliteration. The attention-based model helps the network focus on relevant parts of the input sequence at each step, improving performance, especially in handling cases where the same input characters have different pronunciations.

## Dataset

The dataset used for this project is the  [Aksharantar](https://drive.google.com/file/d/1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A/view) dataset available on Kaggle. It contains parallel text data for various languages, making it suitable for training and evaluating the transliteration model.

## Model Architecture

The model employs an encoder-decoder architecture :
- **Encoder**: Processes the input sequence and encodes it into a fixed-size context vector.
- **Decoder**: Generates the output sequence step-by-step using the context vector provided by the encoder.

### Key Components

- **Embedding Layer**: Converts input characters into dense vectors of fixed size.
- **RNN Layers**: Both encoder and decoder use RNNs (LSTM/GRU) to capture the sequential dependencies.
- **Attention Mechanism**: Allows the decoder to focus on relevant parts of the input sequence during each decoding step.

## Training Strategies

Several strategies were employed to optimize the training process:

1. **Teacher Forcing**: This technique was used to accelerate training by providing the correct output sequence at each training step with a probability of 0.5. This helped the model learn faster and reduced the number of required epochs.
   
2. **Bayesian Hyperparameter Optimization**: A Bayesian Sweep was used to systematically explore and find the best hyperparameters, enhancing the model's character-level accuracy more efficiently than random search methods.

3. **Embedding and Hidden Sizes**: A smaller embedding size was chosen due to the language's limited character set, and larger hidden sizes were used to improve the model's capacity and performance.

4. **Epoch Limitation**: Training was limited to 5 epochs as most models showed minimal improvement beyond this point, significantly reducing the overall training time.

## Hyperparameter Configuration

The best configuration of hyperparameters found for this model is as follows:


input_size = 30  # Number of Latin characters


output_size = 70  # Number of Devanagari characters


embed_size = 128


hidden_size = 256


encoder_layers = 2


decoder_layers = 3


cell_type = 'lstm'


batch_size = 64


num_epochs = 11


drop_prob = 0.2


learning_rate = 0.001


bidirectional = True


## Results

The basic seq2seq model was able to perform transliteration with a reasonable degree of accuracy. However, it struggled in scenarios where the same input characters had different pronunciations due to the lack of an attention mechanism.

```markdown
# Seq2Seq Transliteration Model (With Attention)
```

The model employs an encoder-decoder architecture with an attention mechanism:
- **Encoder**: Processes the input sequence and encodes it into a fixed-size context vector.
- **Decoder**: Generates the output sequence step-by-step, attending to different parts of the input sequence at each timestep.

  
The best configuration of hyperparameters found for this attention based model is as follows:


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


bidirectional = True


## Usage

To run the model, follow these steps:

1. Clone this repository:
   ```sh
   git clone https://github.com/ABANISINGHA/DL_Assignment_3.git
  
2. Install the required dependencies:
  ```sh
   pip install -r requirements.txt
```

## Two ways to use or train
 - By command line arguments using train.py
 - By using the dl-rnn.ipynb (recommended)

### 1) Command line method for training using train.py
Here we need to run train.py file and specify wandb project names and key. But this method will take lots of time if computer doesn't have GPU support. So recommended option is using .ipynb file. The details for training are given below : 

### Usage
First make sure the above dependencies have been installed in your system. Then to use this project, simply clone the repository and run the train.py file. To download the repository, type the below command in the command line interface of git:

`git clone https://github.com/ABANISINGHA/DL_Assignment_3.git`

After running the file as shown above, training and testing accuracies will be printed on the terminal.

Also on using wandb, it is needed to put the corresponding key of the user so it can be logged on to the users wandb project.
It can be changed by searching for `key = <your key>`. at starting of train.py, the key variable will be present.


### 2) Using the dl_a3_seq2seq.ipynb
Here open the dl_a3_seq2seq.ipynb file on colab or kaggle . Then upload the `aksharntar` dataset and preprocess the data. 


`PREPROCESSING STEP` It contains all the preprocessing steps, creating english-index dictionaries, bangla-index dictionaries, then loading data, converting each character into index format to a max sequence length by padding and specifying batch sizes for data. you can change batch size, add different dataset you have to do it here.

`ENCODER MODULE` Here I have defined the encoder class, all the logic of encoder, how processing happens in encoder etc. 

`DECODER MODULE` Here I have defined the decoder class, all the logic of decoder, how processing happens in decoder etc. 

`SEQUENCE TO SEQUENCE MODULE` Here the overall transliteration happens. It will call encoders and decoders and transliterate.

`TRAINING FOR 1 EXAMPLE CONFIGURATION` Here if you wish to run 1 example and see how training goes without logging into wandb, then run this code. 
```python
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
bidirectional = True

model = Seq2Seq(input_size, output_size, hidden_size,embed_size, encoder_layers,decoder_layers,drop_prob, cell_type, bidirectional)
print(model)
```
By changing the values, we can create new configuration and run on it.


`WANDB RUN FUNCTION` For running wandb, this function will be called, it will return list of accruacies and losses for training and validation, which will be used by wandb to log. This run fucntion is same as training function specified above, but mainly suited for logging in wandb.

`WANDB SWEEPS TRAIN FUNCTION` Here wandb sweeps train function is defined, which is being used for sweeping.

`WANDB SWEEP CONFIGURATIONS AND RUN SWEEPS` For specifying the sweep configurations, change the default values of the sweep as you like:
```python
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
```

`TEST ACCURACY CALCULATIONS` Here just pass your test data and it will find the test accuracy of your model - need to find model first, then only run this. It will also log into your wandb project.

`STORE SAMPLE OUTPUT FOR TEST DATA` If you want to store the prediction into a csv file, run these code. If you are in kaggale in output section you can get the csv file.

`RUN SINGLE EXAMPLE AND LOG TO WANDB` Here this is a running example and log all the data into wandb, run these code. project name should be appropriately given.
```python
wandb.login(key = KEY)
wandb.init(project = 'DL_Assignment_3')
model = Seq2Seq(input_size=30,
 output_size=70,
 hidden_size=wandb.config.hidden_layer_size,
embed_size=wandb.config.embedding_size,
encoder_layers=wandb.config.encoder_layers,
decoder_layers=wandb.config.decoder_layers,
drop_prob=wandb.config.dropout,
 cell_type=wandb.config.cell_type,
bidirectional=wandb.config.bidirectional)

```
change these upto your choice of hyperparameter configuration. 

Wandb report : [Link](https://drive.google.com/file/d/1itJ6mbFrJutmfYwU_EjiQ89LiM_uRzD5/view?usp=sharing)


