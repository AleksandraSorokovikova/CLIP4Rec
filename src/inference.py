import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.models import *


class Inference:
    def __init__(
            self,
            film_encoder_path,  # Path to the film encoder model
            text_encoder_path,  # Path to the text encoder model
            vocab,  # Vocabulary object
            dim,  # Dimension of the embeddings
            movies_metadata,  # DataFrame with the movies metadata
            seq_len,  # Movies sequence length
            bert_model,  # Bert embeddings model
            device='cpu',
            hidden_dim=None  # Hidden dimension for the LSTMFilmEncoder model if it's used
    ):

        if hidden_dim is not None:
            self.film_encoder = LSTMFilmEncoder(
                len(vocab.word_to_index), embedding_dim=dim, hidden_dim=hidden_dim
            ).to(device)
        else:
            self.film_encoder = SASFilmEncoder(
                item_num=len(vocab.word_to_index), seq_len=seq_len, embed_dim=dim, device=device
            ).to(device)
        self.text_encoder = TextEncoder(bert_model, output_dim=dim).to(device)
        self.film_encoder.load_state_dict(torch.load(film_encoder_path))
        self.text_encoder.load_state_dict(torch.load(text_encoder_path))

        self.vocab = vocab
        self.movies_metadata = movies_metadata
        self.movie_to_idx = self.movies_metadata.set_index('title')['movieId']
        self.idx_to_movie = self.movies_metadata.set_index('movieId')['title']
        self.seq_len = seq_len
        self.device = device
        self.dim = dim
        self.annoy_model = None

        self.film_encoder.eval()
        self.text_encoder.eval()

    def predict_next_movie(self, list_movies, top_n=10):
        sequence = [self.movie_to_idx[movie] for movie in list_movies]
        input_indices = [self.vocab.word_to_index(word) for word in sequence]
        input_indices = [0] * (self.seq_len - len(input_indices)) + input_indices

        positions = torch.arange(self.seq_len).unsqueeze(0).expand(1, self.seq_len).to(self.device)
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.film_encoder(input_tensor, positions)
            top_logits, top_indices = logits.topk(top_n, dim=1)

        top_words = [self.vocab.index_to_word(idx.item()) for idx in top_indices[0]]
        predicted_movie = [self.idx_to_movie[i] for i in top_words]

        return predicted_movie

    def get_embeddings(self, film_descriptions_encoded, batch_size):

        film_embeddings = {}
        text_embeddings = {}
        all_film_indices = list(self.vocab.index_to_word.keys())
        for i in tqdm(range(0, len(all_film_indices), batch_size)):
            film_ids = []
            descriptions = []
            attention_masks = []
            for j in range(i, min(i + batch_size, len(all_film_indices))):
                film_id = torch.tensor([all_film_indices[j]], dtype=torch.long).to(self.device)
                film_id = film_id.unsqueeze(0)
                film_ids.append(film_id)

                encoded_description = film_descriptions_encoded[all_film_indices[j]]
                description = encoded_description['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = encoded_description['attention_mask'].unsqueeze(0).to(self.device)
                descriptions.append(description)
                attention_masks.append(attention_mask)

            film_ids = torch.cat(film_ids, dim=0)
            descriptions = torch.cat(descriptions, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            film_logits, film_embedding = self.film_encoder(film_ids)
            text_embedding = self.text_encoder(descriptions, attention_masks)

            for film, film_emb, text_emb in zip(film_ids, film_embedding, text_embedding):
                film_embeddings[self.vocab.index_to_word[film.item()]] = film_emb
                text_embeddings[self.vocab.index_to_word[film.item()]] = text_emb

        return film_embeddings, text_embeddings

    def build_annoy_model(self, film_descriptions_encoded, num_trees=10, batch_size=32, search_type='euclidian'):
        print('Counting embeddings...')
        film_embeddings, text_embeddings = self.get_embeddings(film_descriptions_encoded, batch_size)
        print('Building Annoy model...')
        self.annoy_model = AnnoySearchEngine(
            film_embeddings,
            text_embeddings,
            self.dim,
            num_trees=num_trees,
            search_type=search_type,
            movieId_to_title=self.idx_to_movie,
        )
        self.annoy_model.build_trees()
