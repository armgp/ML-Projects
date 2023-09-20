import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pickle
import warnings
warnings.filterwarnings("ignore")


# Load the saved word2idx dictionary
with open('word2idx_nli.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)

# Train the model on GPU

new_embeddings = torch.load('new_embeddings_nli.pt')

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(new_embeddings, freeze=True, sparse=True)
        self.hidden_size = embedding_dim//2
        self.lstm1 = nn.LSTM(input_size=embedding_dim,
                             hidden_size=embedding_dim//2,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True,
                             dropout=dropout
                            )
        self.lstm2 = nn.LSTM(input_size=embedding_dim,
                             hidden_size=embedding_dim//2,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=True,
                             dropout=dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.gamma = nn.Parameter(torch.ones(1))
        self.weights = [0.2, 0.4, 0.4]

    def forward(self, x):
        embeds = self.embedding(x)  # shape: (batch_size, max_seq_length, embedding_dim)

        lstm1_out, _ = self.lstm1(embeds)  # shape: (batch_size, max_seq_length, hidden_size*2)
        lstm2_out, _ = self.lstm2(lstm1_out)  # shape: (batch_size, max_seq_length, hidden_size*2)
        output = self.fc(lstm2_out)  # shape: (batch_size, max_seq_length, vocab_size)
        
        elmo_embeddings = self.gamma * (self.weights[0]*embeds + self.weights[1]*lstm1_out + self.weights[2]*lstm2_out) # shape: (batch_size, max_seq_length, 2*hidden_size)

        return output, elmo_embeddings

VOCAB_SIZE = word2idx['<pad>']+1
EMBEDDING_DIM = 100
DROPOUT = 0.2
model = ELMo(VOCAB_SIZE, EMBEDDING_DIM, DROPOUT)
# model.load_state_dict(torch.load('elmo_model_sst.pt'))
model.load_state_dict(torch.load('elmo_model_nli.pt', map_location=torch.device('cpu')))


import torch
from nltk.tokenize import word_tokenize

# Define a function to convert a sentence to a tensor of word indices
def sentence_to_tensor(sentence, word2idx):
    tokens = word_tokenize(sentence.lower())
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    tensor = torch.LongTensor(indices).unsqueeze(0)  # add batch dimension
    return tensor

# Define a function to get the ELMo embeddings of a sentence
def get_elmo_embeddings(sentence, model, word2idx):
    with torch.no_grad():
        # Convert the sentence to a tensor of word indices
        inputs = sentence_to_tensor(sentence, word2idx)
        
        # Get the ELMo embeddings
        _, elmo_embeddings = model(inputs)
        
        # Remove the batch dimension
        elmo_embeddings = elmo_embeddings.squeeze(0)
        
        return elmo_embeddings
    
# Define a function to convert a sentence to a tensor of word indices
def sentence_to_tensor_padded(sentence, word2idx, pad):
    tokens = word_tokenize(sentence.lower())
    while(len(tokens)<pad):
        tokens.append('<pad>')
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    tensor = torch.LongTensor(indices).unsqueeze(0)  # add batch dimension
    return tensor
    
# Define a function to get the ELMo embeddings of a sentence
def get_elmo_embeddings_padded(sentence, model, word2idx, pad):
    with torch.no_grad():
        # Convert the sentence to a tensor of word indices
        inputs = sentence_to_tensor_padded(sentence, word2idx, pad)
        
        # Get the ELMo embeddings
        _, elmo_embeddings = model(inputs)
        
        # Remove the batch dimension
        elmo_embeddings = elmo_embeddings.squeeze(0)
        
        return elmo_embeddings

class NLIModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NLIModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 4, 1)
        self.fc = nn.Linear(hidden_dim * 4, 3)

    def forward(self, x):
        premise_emb = x[0].float()
        hypothesis_emb = x[1].float() 

        _, (premise_hidden, _) = self.lstm(premise_emb)
        _, (hypothesis_hidden, _) = self.lstm(hypothesis_emb)

        premise_hidden = torch.cat([premise_hidden[-2], premise_hidden[-1]], dim=1)
        hypothesis_hidden = torch.cat([hypothesis_hidden[-2], hypothesis_hidden[-1]], dim=1)

        attention_input = torch.cat([premise_hidden, hypothesis_hidden], dim=1)
        attention_logits = self.attention(attention_input)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attended_input = attention_weights * attention_input

        output = self.fc(attended_input)

        return output
    

hidden_size = 512
num_layers = 2
dropout = 0.2
input_size = 100
output_size = 1

nli_model = NLIModel(VOCAB_SIZE , 100, hidden_size)

nli_model.load_state_dict(torch.load('model_nli.pt',  map_location=torch.device('cpu')))

def getCategory(premise, hypothesis):
    pndh = [get_elmo_embeddings_padded(premise, model, word2idx, 100).float().reshape(1, 100, 100), get_elmo_embeddings_padded(hypothesis, model, word2idx, 100).float().reshape(1, 100, 100)]
    out = nli_model(pndh)
    return torch.argmax(out)

premise = input("Enter premise: ")
hypothesis = input("Enter hypothesis: ")
ans = getCategory(premise, hypothesis).item()
if ans==0:
    print("Entailment")
elif ans==1:
    print("Neutral")
else: 
    print("Contradiction")