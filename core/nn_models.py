import torch
import torch.nn as nn 
from torch.utils.data import Dataset

class LSTM_Classifier(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 lstm_units: int, 
                 num_layers:int = 1, 
                 dropout_rate: float = 0.1,
                 recurrent_dropout_rate: float = 0.2, 
                 pad_token_id=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 
                                      embedding_dim,
                                      padding_idx=pad_token_id
        )        

        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=recurrent_dropout_rate if num_layers > 1 else 0
        ) 

        self.out = nn.Linear(2 * lstm_units, num_classes)

        self.pad_token_id = pad_token_id
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Инициализация bias forget gate единицами
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        lengths = attention_mask.sum(dim=1)

        # if (lengths == 0).any():
        #     raise ValueError(
        #         "Найдены пустые последовательности (состоящие только из PAD). "
        #         "Отфильтруйте их или обеспечьте хотя бы один непад-токен."
        #     )

        lengths = torch.clamp(lengths, min=1)

        x = self.embedding(X)
        x = self.dropout(x)
        
        output, (hidden, _) = self.lstm(x)

        # Конкатенация прямого и обратного состояний последнего слоя
        output = output[torch.arange(output.size(0)), lengths - 1, :]

        logits = self.out(output)

        return logits

class RNN_Text_Classifier(nn.Module):
    def __init__(
            self, 
            num_classes: int,
            vocab_size: int,
            embedding_dim: int,
            hidden_size: int,  
            num_layers: int = 1,
            drop_out_rate: float = 0.1,
    ):
        super().__init__()

        self.embending = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(drop_out_rate)
        self.rnn = nn.RNN(embedding_dim, 
                          hidden_size, 
                          num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_out_rate if num_layers > 1 else 0)
        self.out = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, X: torch.Tensor,  attention_mask: torch.Tensor):
        lengths = attention_mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        output, _ = self.rnn(self.dropout(self.embending(X)))

        output = output[torch.arange(output.size(0)), lengths - 1, :]
        
        logits = self.out(output)

        return logits


class TextDataset(Dataset):
    def __init__(
            self, 
            ids, 
            masks,
            labels,
    ):
        super().__init__()
        self.ids = ids
        self.masks = masks
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return self.ids.shape[0]
    
    def __getitem__(self, idx):

        return {
            "input_ids":        self.ids[idx],
            "attention_mask":   self.masks[idx],
            "labels":           self.labels[idx],
        }
    

if __name__ == "__main__":
    print('ok')