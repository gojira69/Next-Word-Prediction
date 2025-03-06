import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import numpy as np
from collections import Counter
import re
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

### CORPUS CLEANING, TOKENIZATION AND BUILDING VOCAB


def cleanPrideAndPrejudice(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    start_idx = text.find("CHAPTER I.")
    if start_idx == -1:
        return text
    text = text[start_idx:]

    end_idx = text.find("Transcriber's note:")
    if end_idx != -1:
        text = text[:end_idx]

    text = re.sub(r"CHAPTER\s+[IVXLCDM]+", "", text)

    return text


def cleanUllyeses(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    first_idx = text.find("— I —")
    if first_idx == -1:
        return text

    second_idx = text.find("— I —", first_idx + 1)
    if second_idx == -1:
        return text
    text = text[second_idx:]

    text = re.sub(r"—\s+[I|II|III]+\s+—", "", text)

    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    end_idx = text.find("Trieste-Zurich-Paris")
    if end_idx != -1:
        text = text[:end_idx]

    return text


def Tokenizer(inputText):
    text = inputText.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"www\S+", "<URL>", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-za-z0-9.-]+\.[a-z]{2,}", "<MAILID>", text)

    text = re.sub(r"[^\@\#\.\w\?\!\s:-]", "", text)
    text = re.sub(f"-", " ", text)
    text = re.sub(r"_", " ", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n*", "", text)
    text = re.sub(r"\.+", ".", text)

    abbreviations = re.findall(r"\b([A-Z]([a-z]){,2}\.)", text)
    if abbreviations:
        abbreviations_set = set((list(zip(*abbreviations))[0]))

        for word in abbreviations_set:
            pattern = r"\b" + re.escape(word)
            text = re.sub(pattern, word.strip("."), text)

    text = re.sub(r"#\w+\b", "<HASHTAG>", text)
    text = re.sub(r"@\w+\b", "<MENTION>", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)

    # Tokenize each sentence into words
    sentences = [
        sentence.strip() for sentence in re.split(r"[.!?:]+", text) if sentence.strip()
    ]
    sentences = [sentence.split() for sentence in sentences]

    return sentences


def trainTestSplit(sentences, testSize):
    random.seed(69)

    testSplit = random.sample(sentences, testSize)
    duplicateTestSplit = testSplit.copy()
    trainSplit = [
        sentence
        for sentence in sentences
        if sentence not in duplicateTestSplit or duplicateTestSplit.remove(sentence)
    ]

    return trainSplit, testSplit


class TextDataset(Dataset):
    def __init__(self, tokenized_sentences):
        self.seq_length = max(len(sentence) for sentence in tokenized_sentences)

        # Flatten list of lists for vocabulary creation
        all_words = [word for sentence in tokenized_sentences for word in sentence]

        # Create vocabulary
        word_counts = Counter(all_words)
        self.vocab = ["<PAD>", "<UNK>"] + list(word_counts.keys())
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Convert sentences to indices and flatten
        self.data = []
        for sentence in tokenized_sentences:
            sentence_indices = [
                self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence
            ]
            self.data.extend(sentence_indices)

    def __len__(self):
        return max(0, len(self.data) - self.seq_length - 1)

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.seq_length]
        target = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(sequence), torch.tensor(target)


class NGramDataset(Dataset):
    def __init__(self, tokenized_sentences, n):
        self.n = n

        # Count word frequencies
        word_counts = Counter(
            [word for sentence in tokenized_sentences for word in sentence]
        )

        # Filter vocabulary based on minimum frequency
        frequent_words = {word for word, count in word_counts.items()}

        # Create vocabulary with special tokens
        self.vocab = ["<PAD>", "<UNK>"] + list(frequent_words)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Create n-gram samples
        self.data = []
        pad_idx = self.word2idx["<PAD>"]
        unk_idx = self.word2idx["<UNK>"]

        for sentence in tokenized_sentences:
            # Convert words to indices, handling OOV
            indices = []
            for word in sentence:
                if word in self.word2idx:
                    indices.append(self.word2idx[word])
                else:
                    indices.append(unk_idx)

            # Add padding at the beginning
            padded = [pad_idx] * (n - 1) + indices

            # Create n-gram samples
            for i in range(len(indices)):
                context = padded[i : i + n - 1]
                target = indices[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)


class LanguageModelTrainer_FFNN:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print(f"Model moved to {device}")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    def predict_next_words(self, input_tokens, dataset, k=5):
        """
        Predict next words given a list of input tokens
        """
        self.model.eval()

        # Handle both string and list input
        if isinstance(input_tokens, str):
            tokens = input_tokens.strip().split()
        else:
            tokens = input_tokens

        # Ensure we have enough context
        if len(tokens) < dataset.n - 1:
            tokens = ["<PAD>"] * (dataset.n - 1 - len(tokens)) + tokens
        elif len(tokens) > dataset.n - 1:
            tokens = tokens[-(dataset.n - 1) :]

        # Convert tokens to indices, handling OOV
        word_indices = []
        for word in tokens:
            if word in dataset.word2idx:
                word_indices.append(dataset.word2idx[word])
            else:
                word_indices.append(dataset.word2idx["<UNK>"])

        with torch.no_grad():
            input_tensor = torch.tensor(word_indices).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)

            probabilities = torch.softmax(output[0], dim=0)
            top_k_probs, top_k_indices = torch.topk(probabilities, k)

            predictions = []
            for prob, idx in zip(
                top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()
            ):
                word = dataset.idx2word[idx]
                predictions.append((word, prob))

        return predictions


class LanguageModelTrainer:
    def __init__(self, model):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print(f"Model moved to {device}")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()

            hidden = self.model.init_hidden(data.size(0))
            output, _ = self.model(data, hidden)

            loss = self.criterion(
                output.reshape(-1, output.size(-1)), target.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    def predict_next_words(self, input_tokens, dataset, k=5):
        """
        Predict next words given a list of input tokens
        Args:
            input_tokens: List of tokens or space-separated string
            dataset: TextDataset instance
            k: Number of top predictions to return
        """
        self.model.eval()

        # Handle both string and list input
        if isinstance(input_tokens, str):
            tokens = input_tokens.strip().split()
        else:
            tokens = input_tokens

        # Convert tokens to indices
        word_indices = [
            dataset.word2idx.get(word, dataset.word2idx["<UNK>"]) for word in tokens
        ]

        # # Trim sequence if longer than seq_length
        # if len(word_indices) > dataset.seq_length:
        #     word_indices = word_indices[-dataset.seq_length:]

        with torch.no_grad():
            input_tensor = torch.tensor(word_indices).unsqueeze(0).to(device)
            hidden = self.model.init_hidden(1)
            output, _ = self.model(input_tensor, hidden)

            probabilities = torch.softmax(output[0, -1], dim=0)
            top_k_probs, top_k_indices = torch.topk(probabilities, k)

            predictions = []
            for prob, idx in zip(
                top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()
            ):
                word = dataset.idx2word[idx]
                predictions.append((word, prob))

        return predictions


class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(FFNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Additional dense layer
        self.output = nn.Linear(hidden_dim // 2, vocab_size)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)  # Helps regularization

    def forward(self, x):
        # Embed all context words: [batch_size, context_size, embedding_dim]
        embedded = self.embedding(x)

        # Flatten the embeddings: [batch_size, context_size * embedding_dim]
        batch_size = embedded.size(0)
        flattened = embedded.view(batch_size, -1)

        # Pass through multiple dense layers
        hidden = self.activation(self.hidden1(flattened))
        hidden = self.dropout(hidden)  # Dropout for regularization
        hidden = self.activation(self.hidden2(hidden))

        output = self.output(hidden)
        return output


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        if hidden is None:
            # LSTM needs both hidden state and cell state
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            hidden = (h0, c0)

        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


def main():
    file_path = input("Enter Corpus Path: ")
    k = int(input("Enter Number of Candidates for Next Word: "))
    lm_type = input("Enter LM Type: ")

    print(f"Corpus Path: {file_path}")
    print(f"Language Model Type: {lm_type}")
    print(f"Number of Candidates: {k}")

    filteredText = ""
    corp = ""

    # corpus cleaning and tokenization
    if "corpus_1.txt" in file_path:
        filteredText = cleanPrideAndPrejudice(file_path)
        corp = "papc"
    elif "corpus_2.txt" in file_path:
        filteredText = cleanUllyeses(file_path)
        corp = "uc"
    else:
        print("Corpus doesn't exist!")
        exit()

    sentences = Tokenizer(filteredText)
    train_text, test_text = trainTestSplit(sentences, 1000)

    vocab = Counter(word for sentence in train_text for word in sentence)
    word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    word_to_idx["<UNK>"] = len(word_to_idx)
    idx_to_word[len(idx_to_word)] = "<UNK>"

    print(f"Using device: {device}")

    if lm_type == "f":
        n = int(input("Enter value of N: "))

        model = torch.load(f"pretrained_models/ffnn_{corp}_{n}.pt", map_location=device)
        model.eval()
        dataset = NGramDataset(train_text, n)
        trainer = LanguageModelTrainer_FFNN(model, device)

        user_input = input("Input sentence: ")
        predictions = trainer.predict_next_words(user_input, dataset, k)

        print("\nTest predictions:")
        for word, prob in predictions:
            print(f"Word: {word}, Probability: {prob:.4f}")

    elif lm_type == "r":

        model = torch.load(f"pretrained_models/rnn_{corp}.pt", map_location=device)
        model.eval()
        dataset = TextDataset(train_text)
        trainer = LanguageModelTrainer(model)

        user_input = input("Input sentence: ")
        predictions = trainer.predict_next_words(user_input, dataset, k)

        print("\nTest predictions:")
        for word, prob in predictions:
            print(f"Word: {word}, Probability: {prob:.4f}")

    elif lm_type == "l":

        model = torch.load(f"pretrained_models/lstm_{corp}.pt", map_location=device)
        model.eval()
        dataset = TextDataset(train_text)
        trainer = LanguageModelTrainer(model)

        user_input = input("Input sentence: ")
        predictions = trainer.predict_next_words(user_input, dataset, k)

        print("\nTest predictions:")
        for word, prob in predictions:
            print(f"Word: {word}, Probability: {prob:.4f}")

    else:
        exit()


if __name__ == "__main__":
    main()
