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
            bert_tokenizer,  # Bert tokenizer
            device='cpu',
            hidden_dim=None,  # Hidden dimension for the LSTMFilmEncoder model if it's used
            max_length=125
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
        self.max_length = max_length
        self.tokenizer = bert_tokenizer
        self.annoy_model = None

        self.film_encoder.eval()
        self.text_encoder.eval()

    def predict_next_movie(self, list_movies, top_n=10):
        sequence = [self.movie_to_idx[movie] for movie in list_movies]
        input_indices = [self.vocab.word_to_idx(word) for word in sequence]
        input_indices = [0] * (self.seq_len - len(input_indices)) + input_indices
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, embeddings, _ = self.film_encoder(input_tensor)
            top_logits, top_indices = logits.topk(top_n, dim=1)

        top_words = [self.vocab.idx_to_word(idx.item()) for idx in top_indices[0]]
        predicted_movie = [self.idx_to_movie[i] for i in top_words]

        return predicted_movie

    def get_embeddings(self, film_descriptions_encoded, batch_size=32):

        film_embeddings = {}
        text_embeddings = {}
        all_film_indices = list(self.vocab.index_to_word.keys())
        for i in tqdm(range(0, len(all_film_indices), batch_size)):
            film_ids = []
            descriptions = []
            attention_masks = []
            for j in range(i, min(i + batch_size, len(all_film_indices))):
                if self.vocab.index_to_word[all_film_indices[j]] == '<PAD>':
                    continue
                film_id = torch.tensor([all_film_indices[j]], dtype=torch.long).to(self.device)
                film_id = film_id.unsqueeze(0)
                film_ids.append(film_id)

                encoded_description = film_descriptions_encoded[all_film_indices[j]]
                description = encoded_description['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = encoded_description['attention_mask'].unsqueeze(0).to(self.device)
                descriptions.append(description)
                attention_masks.append(attention_mask)

            film_ids = torch.cat(film_ids, dim=0)
            descriptions = torch.cat(descriptions, dim=0).unsqueeze(dim=1)
            attention_masks = torch.cat(attention_masks, dim=0).unsqueeze(dim=1)

            with torch.no_grad():
                film_logits, film_embedding, _ = self.film_encoder(film_ids)
                film_embedding = film_embedding.squeeze(dim=1)
                text_embedding = self.text_encoder(descriptions, attention_masks)
                text_embedding = text_embedding.squeeze(dim=1)

            for film, film_emb, text_emb in zip(film_ids, film_embedding, text_embedding):
                film_embeddings[self.vocab.index_to_word[film.item()]] = film_emb
                text_embeddings[self.vocab.index_to_word[film.item()]] = text_emb

        return film_embeddings, text_embeddings

    def init_annoy_model(self, text_index_path, film_index_path, idx_to_movieId_path, num_trees=10, search_type='euclidean'):
        self.annoy_model = AnnoySearchEngine(
            dim=self.dim,
            num_trees=num_trees,
            search_type=search_type,
        )
        self.annoy_model.init_index(text_index_path, film_index_path, idx_to_movieId_path)

    def get_film_embeddings(self, film_ids):
        return_list = True
        if type(film_ids) == int:
            film_ids = [film_ids]
            return_list = False
        indices = [self.vocab.word_to_index[film_id] for film_id in film_ids]
        film_ids_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(dim=1).to(self.device)
        with torch.no_grad():
            film_logits, film_embeddings, _ = self.film_encoder(film_ids_tensor)
        if not return_list:
            return film_embeddings.squeeze(dim=1)[0]
        return film_embeddings.squeeze(dim=1)

    def get_text_embeddings(self, descriptions):
        return_list = True
        if type(descriptions) == str:
            descriptions = [descriptions]
            return_list = False
        encoded_descriptions = self.tokenizer(descriptions, return_tensors="pt", max_length=self.max_length,
                                              truncation=True, padding="max_length")
        input_ids = encoded_descriptions['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoded_descriptions['attention_mask'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids, attention_mask)
        if not return_list:
            return text_embeddings.squeeze(dim=0)[0]
        return text_embeddings.squeeze(dim=0)

    def get_mean_embedding(self, film_ids):
        film_embeddings = self.get_film_embeddings(film_ids)
        return film_embeddings.mean(dim=0)

    def get_agreggated_embedding(self, film_ids):
        indices = [self.vocab.word_to_index[film_id] for film_id in film_ids]
        indices = [0] * (self.seq_len - len(indices)) + indices
        film_ids_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            film_logits, film_embeddings, aggregated_embedding = self.film_encoder(film_ids_tensor)
        return aggregated_embedding.squeeze(dim=0)

    def search_text(self, text, in_films=False, top_n=10):
        if self.annoy_model is None:
            raise ValueError('Annoy model is not built. Run build_annoy_model method first.')
        text_embedding = self.get_text_embeddings(text).cpu().numpy()
        if in_films:
            result = self.annoy_model.search_in_film_index(text_embedding, top_n)
        else:
            result = self.annoy_model.search_in_text_index(text_embedding, top_n)
        return [self.idx_to_movie[i] for i in result]

    def search_film(self, film_id, in_texts=False, top_n=10):
        if self.annoy_model is None:
            raise ValueError('Annoy model is not built. Run build_annoy_model method first.')
        film_embedding = self.get_film_embeddings(film_id).cpu().numpy()
        if in_texts:
            result = self.annoy_model.search_in_text_index(film_embedding, top_n)
        else:
            result = self.annoy_model.search_in_film_index(film_embedding, top_n)
        return [self.idx_to_movie[i] for i in result]

    def search_film_by_sequence_and_text(self, film_ids, text, in_films=True, agg=True, top_n=10, ration=0.5):
        if self.annoy_model is None:
            raise ValueError('Annoy model is not built. Run build_annoy_model method first.')
        if agg:
            agg_emb = self.get_agreggated_embedding(film_ids).cpu().numpy()
        else:
            agg_emb = self.get_film_embeddings(film_ids).mean(dim=0).cpu().numpy()

        text_emb = self.get_text_embeddings(text).cpu().numpy()

        agg_emb = agg_emb / np.linalg.norm(agg_emb, ord=2, axis=-1, keepdims=True)
        text_emb = text_emb / np.linalg.norm(text_emb, ord=2, axis=-1, keepdims=True)

        mean_emb = (1 - ration) * agg_emb + ration * text_emb
        assert mean_emb.shape[0] == self.dim

        if in_films:
            result = self.annoy_model.search_in_film_index(mean_emb, top_n)
        else:
            result = self.annoy_model.search_in_text_index(mean_emb, top_n)
        return [self.idx_to_movie[i] for i in result]
