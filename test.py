from bpe import BpeTokenizer
from unigram import UnigramTokenizer

def run_bbpe_test():
    # A corpus with Unicode characters and spaces
    corpus = "Hello world! 👋 The quick brown fox jumped over the lazy dog. 你好世界！"
    vocab_size = 500  # A larger vocabulary to see more meaningful merges

    tokenizer = BpeTokenizer()
    print("--- Training the tokenizer (BBPE) ---")
    tokenizer.train(corpus, vocab_size)

    print("\n--- Encoding and Decoding ---")
    new_text = "The world is wonderful. 世界是美好的。"
    
    encoded_ids = tokenizer.encode(new_text)
    print(f"Original text: '{new_text}'")
    print(f"Encoded IDs: {encoded_ids}")
    
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    # Assert for lossless conversion
    assert new_text == decoded_text
    print(f"\nIs the conversion lossless? {new_text == decoded_text} ✅")


def run_unigram_test():
    corpus = "The quick brown fox jumped over the lazy dog."
    vocab_size = 50

    tokenizer = UnigramTokenizer()
    print("--- Training the tokenizer (Unigram) ---")
    tokenizer.train(corpus, vocab_size)

    print("\n--- Encoding and Decoding ---")
    new_text = "The quick brown fox."
    
    encoded_ids = tokenizer.encode(new_text)
    print(f"Original text: '{new_text}'")
    print(f"Encoded IDs: {encoded_ids}")
    
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    assert new_text == decoded_text
    print(f"\nIs the conversion lossless? {new_text == decoded_text} ✅")

    # Example of saving and loading
    tokenizer.save('unigram_tokenizer.pkl')
    loaded_tokenizer = UnigramTokenizer()
    loaded_tokenizer.load('unigram_tokenizer.pkl')
    print("\nTokenizer saved and loaded successfully!")

if __name__ == "__main__":
    run_bbpe_test()
    print("\n" + "="*50 + "\n")
    run_unigram_test()
