import collections
import math
import pickle
import unicodedata

class UnigramTokenizer:
    def __init__(self):
        self.vocab = {}
        self.scores = {}
        self.inverse_vocab = {}
        self.special_space = ' '

    def _preprocess_text(self, text):
        return text.replace(' ', self.special_space)

    def _postprocess_text(self, text):
        return text.replace(self.special_space, ' ')

    def _viterbi_segment(self, text):
        """
        Implements the Viterbi algorithm to find the best segmentation.
        Returns a tuple of (list_of_tokens, total_score).
        """
        text_bytes = text.encode('utf-8')
        dp = [(0.0, 0)] * (len(text_bytes) + 1)
        
        # Initialize dp[0] with a zero score, representing the start
        dp[0] = (0.0, 0)
        
        for i in range(1, len(text_bytes) + 1):
            max_score = -1e10
            best_prev_j = 0
            
            # Iterate through all possible subwords ending at position i
            for j in range(i):
                subword = text_bytes[j:i]
                if subword in self.scores:
                    score = self.scores[subword]
                    new_score = dp[j][0] + score
                    if new_score > max_score:
                        max_score = new_score
                        best_prev_j = j
            dp[i] = (max_score, best_prev_j)
        
        # Reconstruct the best path by backtracking
        tokens = []
        i = len(text_bytes)
        while i > 0:
            j = dp[i][1]
            token = text_bytes[j:i]
            tokens.append(token)
            i = j
        
        tokens.reverse()
        return (tokens, dp[len(text_bytes)][0])

    def train(self, corpus, vocab_size):
        preprocessed_corpus = self._preprocess_text(corpus)
        text_bytes = preprocessed_corpus.encode('utf-8')
        
        # Initial vocabulary generation (same as before)
        initial_subwords = set()
        for i in range(len(text_bytes)):
            for j in range(i + 1, min(i + 15, len(text_bytes) + 1)):
                initial_subwords.add(text_bytes[i:j])
        
        # Initialize scores based on simple frequency
        counts = collections.Counter(initial_subwords)
        total_count = sum(counts.values())
        self.scores = {sub: math.log(count / total_count) for sub, count in counts.items()}
        
        # Pruning loop: repeatedly remove the least useful tokens
        while len(self.scores) > vocab_size:
            # E-Step: Segment the corpus using the current scores
            tokens, _ = self._viterbi_segment(preprocessed_corpus)
            
            # Count the frequency of each token
            token_counts = collections.Counter(tokens)
            
            # Sort tokens by frequency in ascending order
            sorted_tokens = sorted(self.scores.keys(), key=lambda t: token_counts.get(t, 0))
            
            # Prune a fixed percentage of tokens, but never prune single characters
            pruned_count = 0
            for token in sorted_tokens:
                if len(self.scores) <= vocab_size:
                    break
                # Only prune multi-character tokens
                if len(token) > 1:
                    del self.scores[token]
                    pruned_count += 1
            
            if pruned_count == 0:
                print("No more tokens to prune. Stopping.")
                break
                
            print(f"Pruned {pruned_count} tokens. New vocab size: {len(self.scores)}")
        
        # Finalize vocabulary
        self.vocab = {sub: i for i, sub in enumerate(self.scores.keys())}
        self.inverse_vocab = {i: sub for sub, i in self.vocab.items()}
        
        print(f"Final Unigram vocabulary size: {len(self.vocab)}")
        
    def encode(self, text):
        tokens, _ = self._viterbi_segment(self._preprocess_text(text))
        return [self.vocab[t] for t in tokens]
        
    def decode(self, token_ids):
        token_bytes = b''
        for i in token_ids:
            token_bytes += self.inverse_vocab[i]
            
        decoded_text = token_bytes.decode('utf-8')
        return self._postprocess_text(decoded_text)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))