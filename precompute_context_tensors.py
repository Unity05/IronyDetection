import torch
import torch.nn as nn
import pandas as pd
import math

from model import ContextModel, PositionalEncoding
from Dataset import IronyClassificationDataset


def collate_fn(batch):
    return tuple(zip(*batch))


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


model_checkpoint_path = 'models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_0.9.pth'
model_state_dict = torch.load(model_checkpoint_path, map_location='cuda')

# print(model_state_dict['model_state_dict'])
word_embedding_layer = {'weight': model_state_dict['model_state_dict']['word_embedding.weight']}
# print(word_embedding_layer)
word_embedding = nn.Embedding(num_embeddings=int(1.0e5), embedding_dim=512).to(device)
word_embedding.load_state_dict(word_embedding_layer)

positional_encoder = PositionalEncoding().to(device)

context_model_layers = {k[18:]: v for k, v in model_state_dict['model_state_dict'].items() if 'context_embedding' in k}

model = ContextModel(
    d_model=512,
    d_context=128
)
model.load_state_dict(context_model_layers)
model.eval()

train_dataset = IronyClassificationDataset(mode='test', top_k=1.0e5, root='data/irony_data')
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


precomputed_context_tensors_list = []

with torch.no_grad():
    for i, data in enumerate(train_dataloader):
        _, _, parent_utterance, parent_utterance_len, _ = data
        parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False,
                                                     padding_value=10002).to(device)
        parent_utterance = word_embedding(parent_utterance.long()) * math.sqrt(512)
        parent_utterance = positional_encoder(parent_utterance)
        parent_utterance_context_tensor = model(word_embedding=parent_utterance, utterance_lengths=parent_utterance_len)

        # print(parent_utterance_context_tensor.shape)
        # print(parent_utterance_context_tensor)

        precomputed_context_tensors_list.append(parent_utterance_context_tensor)

        # exit(-1)

precomputed_context_tensors_tensor = torch.stack(precomputed_context_tensors_list)
torch.save(precomputed_context_tensors_tensor, 'data/irony_data/precomputed_context_tensors_tensor.pth')


# df = pd.read_csv('data/irony_data/train-balanced-sarcasm-train.csv')
