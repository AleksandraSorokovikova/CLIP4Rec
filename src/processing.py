import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


seq_len = 5


def create_ratings_df(
        frac=None,
        number_of_movies=None,
        links_path='archive/links.csv',
        movies_metadata_path='archive/movies_metadata.csv',
        ratings_path='archive/ratings.csv'
):
    links = pd.read_csv(links_path)
    movies_metadata = pd.read_csv(movies_metadata_path)
    movies_metadata['popularity'] = movies_metadata['popularity'].fillna(0)
    movies_metadata = movies_metadata.dropna(subset=['overview'])
    ratings = pd.read_csv(ratings_path)

    if number_of_movies is not None:
        movies_metadata['release_date'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce')
        movies_metadata['release_year'] = movies_metadata['release_date'].dt.year
        movies_metadata = movies_metadata[movies_metadata['release_year'].notnull()]
        movies_metadata['release_year'] = movies_metadata['release_year'].astype(int)
        movies_metadata = movies_metadata.query('release_year >= 2000')
        movies_metadata = movies_metadata[movies_metadata['imdb_id'].notnull()]
        movies_metadata = movies_metadata.drop_duplicates(subset='imdb_id')
        movies_metadata['popularity'] = movies_metadata['popularity'].fillna(0).astype(float)
        movies_metadata = movies_metadata[movies_metadata['overview'].notnull()]
        movies_metadata = movies_metadata.query('original_language == "en"')
        movies_metadata = movies_metadata.sort_values('popularity', ascending=False).head(number_of_movies)

    if frac is not None:
        user_ids = ratings['userId'].unique()
        np.random.shuffle(user_ids)
        user_ids = user_ids[:int(len(user_ids) * frac)]
        ratings = ratings[ratings['userId'].isin(user_ids)]

    links['tmdbId'] = links['tmdbId'].fillna(0).astype(int)
    links = links.rename(columns={'tmdbId': 'id'})
    links = links.drop_duplicates(subset='id', keep='first')
    links['id'] = links['id'].astype(str)

    movies_metadata['id'] = movies_metadata['id'].astype(str, errors='ignore')
    movies_metadata = movies_metadata.drop_duplicates('id')
    movies_metadata['movieId'] = movies_metadata['id'].map(links.set_index('id')['movieId']).fillna(0).astype(int)

    def convert(row):
      try:
        return float(row)
      except:
        return 0.0

    movies_metadata['popularity'] = movies_metadata['popularity'].apply(convert)
    movie_descriptions = movies_metadata.set_index('movieId')['overview']
    ratings_df = pd.merge(ratings, movies_metadata[['movieId', 'title']])[['userId', 'movieId', 'title', 'rating', 'timestamp']]
    ratings_df = ratings_df.dropna()
    ratings_df = ratings_df.sort_values('timestamp')
    ratings_df = ratings_df.query('rating >= 3')

    return ratings_df, movie_descriptions, movies_metadata


def split_sequences(seq, max_len=seq_len + 1):
    n = len(seq)
    if n <= max_len:
        return [seq]
    else:
        for size in range(max_len, 1, -1):
            if n % size >= 2 or n % size == 0:
                break
        split_sizes = [size] * (n // size) + ([n % size] if n % size != 0 else [])
        if split_sizes[-1] == 1:
            split_sizes[-2] += 1
            split_sizes.pop()
        return [seq[i:i + size] for i, size in zip(np.cumsum([0] + split_sizes[:-1]), split_sizes)]


def get_sequences(ratings_df):
    filtered_df = ratings_df.groupby('userId').filter(lambda x: len(x) >= 3).sort_values(by=['timestamp'])
    grouped = filtered_df.groupby('userId')['movieId'].apply(list)
    result = grouped.apply(split_sequences)

    users_to_sequences_id = {}
    sequences_id_to_users = {}
    sequences = []
    for user, seqs in result.items():
        for seq in seqs:
            if len(seq) < 3:
                continue
            sequences.append(seq)
            if user in users_to_sequences_id:
                users_to_sequences_id[user].append(len(sequences) - 1)
            else:
                users_to_sequences_id[user] = [len(sequences) - 1]
            sequences_id_to_users[len(sequences) - 1] = user

    return sequences


class Vocabulary:
    def __init__(self, pad_token="<PAD>"):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_count = Counter()
        self.pad_token = pad_token
        self.init_vocab()
        self.vocab_size = len(self.word_to_index)

    def init_vocab(self):
        self.add_word(self.pad_token)

    def add_word(self, word):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            self.word_count[word] += 1

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.add_word(word)

    def word_to_idx(self, word):
        return self.word_to_index.get(word, self.word_to_index[self.pad_token])

    def idx_to_word(self, idx):
        return self.index_to_word[idx]


def prepare_dataset(user_item_sequences, film_descriptions, tokenizer, vocab, max_seq_length=seq_len, encode_descriptions=False):

    film_ids_seq = []
    target_id_seq = []
    for idx in tqdm(range(len(user_item_sequences))):
        film_sequence = [vocab.word_to_idx(i) for i in user_item_sequences[idx][:-1]]
        target_index = vocab.word_to_idx(user_item_sequences[idx][-1])
        film_sequence = [0] * (max_seq_length - len(film_sequence)) + film_sequence

        film_ids_seq.append(film_sequence)
        target_id_seq.append(target_index)

    data = {
        'film_ids': film_ids_seq,
        'target_id': target_id_seq
    }

    if encode_descriptions:
        film_descriptions_encoded = {}
        for film_id in tqdm(vocab.index_to_word):
            if vocab.index_to_word[film_id] in film_descriptions:
                description = film_descriptions[vocab.index_to_word[film_id]]
            else:
                description = ""
            encoded_description = tokenizer(description, return_tensors="pt", max_length=125, truncation=True, padding="max_length")
            film_descriptions_encoded[film_id] = {
                'input_ids': encoded_description['input_ids'].squeeze(0),
                'attention_mask': encoded_description['attention_mask'].squeeze(0)
            }
        return data, film_descriptions_encoded
    else:
        return data


class FilmRecommendationDataset(Dataset):
    def __init__(self, data, film_descriptions_encoded):

        self.film_ids = data['film_ids']
        self.target_id = data['target_id']
        self.film_descriptions_encoded = film_descriptions_encoded

    def __len__(self):
        return len(self.film_ids)

    def __getitem__(self, idx):
        descriptions = []
        attention_masks = []
        film_ids = self.film_ids[idx]
        for film_id in film_ids:
            descriptions.append(self.film_descriptions_encoded[film_id]['input_ids'])
            attention_masks.append(self.film_descriptions_encoded[film_id]['attention_mask'])
        descriptions = torch.stack(descriptions)
        attention_masks = torch.stack(attention_masks)
        return {
            'film_ids': torch.tensor(film_ids, dtype=torch.long),
            'target_id': self.target_id[idx],
            'descriptions': descriptions,
            'attention_masks': attention_masks
        }


class MoviesDataset(Dataset):
    def __init__(self, sequences, vocab):
        self.sequences = sequences
        self.vocab = vocab

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        words = self.sequences[idx]
        input_indices = [self.vocab.word_to_idx(word) for word in words[:-1]]
        target_index = self.vocab.word_to_idx(words[-1])
        return input_indices, target_index


class CollateFunction:
    def __init__(self, pad_idx=0, max_length=seq_len):
        self.pad_idx = pad_idx
        self.max_length = max_length

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        inputs = [[self.pad_idx] * (self.max_length - len(inp)) + inp for inp in inputs]

        inputs_padded = torch.tensor(inputs, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        return inputs_padded, targets_tensor
