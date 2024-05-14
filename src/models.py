import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm


class TextEncoder(nn.Module):
    def __init__(self, bert_model, output_dim):
        super(TextEncoder, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, outputs=None):
        batch_size, num_films, seq_len = input_ids.size()

        if outputs is None:
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        fc_output = self.fc(cls_embedding)
        activated_output = self.tanh(fc_output)

        activated_output = activated_output.view(batch_size, num_films, -1)
        return activated_output


class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.aggregate_fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, hidden, encoder_outputs, mask):
        hidden = hidden.unsqueeze(2)
        energy = torch.bmm(encoder_outputs, hidden).squeeze(2)
        energy = energy.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(energy)
        context = torch.bmm(encoder_outputs.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        aggregated_embedding = self.aggregate_fc(context)
        return context, attention_weights, aggregated_embedding


class LSTMFilmEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMFilmEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.attention = Attention(embedding_dim, hidden_dim)
        self.batch_norm_fc = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.batch_norm(output.contiguous().view(-1, output.shape[2])).view(output.shape)
        mask = (x != 0)
        context, attention_weights, aggregated_embedding = self.attention(hidden[-1], output, mask)
        context = self.batch_norm_fc(context)
        logits = self.fc(context)
        return logits, embedded, aggregated_embedding


class SASFilmEncoder(nn.Module):
    def __init__(self, item_num, embed_dim, seq_len, device, num_blocks=4, num_heads=2, dropout_rate=0.1):
        super(SASFilmEncoder, self).__init__()
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.device = device

        self.item_embedding = nn.Embedding(item_num, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_layer = nn.Linear(embed_dim, item_num)
        self.aggregate_fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        embeddings = self.item_embedding(x)
        positions = torch.arange(self.seq_len).unsqueeze(0).expand(len(x), self.seq_len).to(self.device)
        x = embeddings + self.position_embedding(positions)
        mask = (x == 0).all(dim=2)

        aggregated_embedding = None

        for block in self.blocks:
            x, block_aggregated_embedding = block(x, mask)
            if aggregated_embedding is None:
                aggregated_embedding = block_aggregated_embedding
            else:
                aggregated_embedding += block_aggregated_embedding

        x = self.final_layer(x)
        aggregated_embedding = self.aggregate_fc(aggregated_embedding)
        return x[:, -1, :], embeddings, aggregated_embedding


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.aggregate_fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layer_norm1(x + attn_output).permute(1, 0, 2)
        ff_output = self.feed_forward(x)
        output = self.layer_norm2(x + ff_output)
        aggregated_embedding = torch.mean(output, dim=1) 
        aggregated_embedding = self.aggregate_fc(aggregated_embedding) 

        return output, aggregated_embedding


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, device='cpu'):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, film_ids, films_embeddings, text_embeddings):
        batch_loss = 0
        for inputs, film_embedding, text_embedding in zip(film_ids, films_embeddings, text_embeddings):
            mask = (inputs != 0).to(self.device)
            film_embedding = film_embedding[mask]
            text_embedding = text_embedding[mask]
            assert films_embeddings.size() == text_embeddings.size()

            film_embedding = F.normalize(film_embedding, p=2, dim=1)
            text_embedding = F.normalize(text_embedding, p=2, dim=1)

            similarity = torch.mm(film_embedding, text_embedding.T) * np.exp(self.temperature)
            labels = torch.arange(similarity.size(0)).type_as(similarity).long()

            loss_i2t = F.cross_entropy(similarity, labels)
            loss_t2i = F.cross_entropy(similarity.t(), labels)
            loss = (loss_i2t + loss_t2i) / 2
            batch_loss += loss
        batch_loss = batch_loss / films_embeddings
        return batch_loss


class AggregatedLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.1, device='cpu'):
        super(AggregatedLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss().to(self.device)
        self.clip_contrastive_loss = CLIPLoss(temperature=temperature).to(self.device)
        (self.ce_mean, self.ce_std, self.clip_mean, self.clip_std) = (0.0, 1.0, 0.0, 1.0)

    def forward(self, film_ids, film_logits, target_id, film_embeddings, text_embeddings, train=True):
        self.classification_loss = self.cross_entropy(film_logits, target_id)
        self.contrastive_loss = self.clip_contrastive_loss(film_ids, film_embeddings, text_embeddings)

        # if train:
        #   new_ce_mean = self.alpha * self.classification_loss.item() + (1 - self.alpha) * self.ce_mean
        #   new_ce_std = self.alpha * (self.classification_loss - self.ce_mean).abs().item() + (1 - self.alpha) * self.ce_std
        #   new_clip_mean = self.alpha * self.contrastive_loss.item() + (1 - self.alpha) * self.clip_mean
        #   new_clip_std = self.alpha * (self.contrastive_loss - self.clip_mean).abs().item() + (1 - self.alpha) * self.clip_std
        #   (self.ce_mean, self.ce_std, self.clip_mean, self.clip_std) = (new_ce_mean, new_ce_std, new_clip_mean, new_clip_std)
        # normalized_ce = (self.classification_loss - self.ce_mean) / self.ce_std
        # normalized_clip = (self.contrastive_loss - self.clip_mean) / self.clip_std

        loss = torch.log(1 + self.classification_loss) + torch.log(1 + self.contrastive_loss)

        return loss


class AnnoySearchEngine:
    def __init__(self, dim, num_trees, search_type):
        self.dim = dim
        self.num_trees = num_trees
        self.search_type = search_type
        self.idx_to_movieId = {}
        self.text_index = None
        self.film_index = None

    def build_trees(self, text_embeddings_dict, film_embeddings_dict):
        self.text_index = AnnoyIndex(self.dim, self.search_type)
        self.film_index = AnnoyIndex(self.dim, self.search_type)

        for i, (film_id, text_embedding) in enumerate(tqdm(text_embeddings_dict.items())):
            self.idx_to_movieId[i] = film_id

            text_embedding = text_embedding.cpu().numpy()
            text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
            
            film_embedding = film_embeddings_dict[film_id].cpu().numpy()
            film_embedding = film_embedding / np.linalg.norm(film_embedding, ord=2, axis=-1, keepdims=True)
            
            self.text_index.add_item(i, text_embedding)
            self.film_index.add_item(i, film_embedding)

        self.text_index.build(self.num_trees)
        self.film_index.build(self.num_trees)

    def search_in_text_index(self, embedding, top_n=10):
        embedding = embedding / np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
        idxs = self.text_index.get_nns_by_vector(embedding, top_n)
        return [self.idx_to_movieId[i] for i in idxs]

    def search_in_film_index(self, embedding, top_n=10):
        embedding = embedding / np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
        idxs = self.film_index.get_nns_by_vector(embedding, top_n)
        return [self.idx_to_movieId[i] for i in idxs]
        