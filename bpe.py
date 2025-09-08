import collections
import re
import unicodedata

class BpeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        # Special Unicode character for whitespace
        self.special_space = ' '

    def get_pair_counts(self, tokens):
        counts = collections.defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            counts[pair] += 1
        return counts

    def train(self, text_corpus, vocab_size):
        # Step 1: Pre-process text and convert to bytes
        preprocessed_text = self._preprocess_text(text_corpus)
        initial_bytes = preprocessed_text.encode('utf-8')
        
        # Step 2: Initialize vocabulary with all 256 byte values
        self.vocab = {bytes([i]): i for i in range(256)}
        self.inverse_vocab = {i: bytes([i]) for i in range(256)}
        
        token_list = [bytes([b]) for b in initial_bytes]
        
        num_merges = vocab_size - len(self.vocab)

        for i in range(num_merges):
            # 3. Find the most frequent pair of byte tokens
            pair_counts = self.get_pair_counts(token_list)
            
            if not pair_counts:
                break
                
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            
            # 4. Create new token and update the corpus
            new_token_id = len(self.vocab)
            new_token_bytes = most_frequent_pair[0] + most_frequent_pair[1]
            
            self.vocab[new_token_bytes] = new_token_id
            self.inverse_vocab[new_token_id] = new_token_bytes
            self.merges[most_frequent_pair] = new_token_id
            
            # Rebuilding the token list by replacing the merged pair
            new_token_list = []
            k = 0
            while k < len(token_list):
                if k < len(token_list) - 1 and (token_list[k], token_list[k+1]) == most_frequent_pair:
                    new_token_list.append(new_token_bytes)
                    k += 2
                else:
                    new_token_list.append(token_list[k])
                    k += 1
            
            token_list = new_token_list
            # The print statement also needs to handle bytes
            # You can decode the bytes to make the output readable
            print(f"Merge {i+1}: {most_frequent_pair} -> {new_token_bytes.decode('utf-8', errors='replace')}")

    def encode(self, text):
        # Step 1: Pre-process text and get initial byte tokens
        preprocessed_text = self._preprocess_text(text)
        token_list = [bytes([b]) for b in preprocessed_text.encode('utf-8')]

        # Step 2: Apply merges in the order they were created
        sorted_merges = sorted(self.merges.keys(), key=lambda x: self.merges[x])
        
        for pair_bytes in sorted_merges:
            # Rebuild the token list by replacing the merged pair
            new_token_list = []
            k = 0
            while k < len(token_list):
                if k < len(token_list) - 1 and (token_list[k], token_list[k+1]) == pair_bytes:
                    new_token_list.append(pair_bytes[0] + pair_bytes[1])
                    k += 2
                else:
                    new_token_list.append(token_list[k])
                    k += 1
            token_list = new_token_list
        
        # Step 3: Convert the final list of byte tokens to integer IDs
        return [self.vocab[t] for t in token_list]
    
    def decode(self, token_ids):
        token_bytes = b''
        for i in token_ids:
            token_bytes += self.inverse_vocab[i]
            
        decoded_text = token_bytes.decode('utf-8')
        
        return self._postprocess_text(decoded_text)
    
    def _preprocess_text(self, text):
        return text.replace(' ', self.special_space)

    def _postprocess_text(self, text):
        return text.replace(self.special_space, ' ')