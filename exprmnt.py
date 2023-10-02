import streamlit as st
import string
import random
import math
from typing import List
from modeler import NgramModel, simple_NgramModel


def tokenize(text: str) -> List[str]:
    # Remove punctuation and split text into tokens
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.lower().split()
    return tokens


def perplexity(ngram_model, test_data):
    """
    Calculate perplexity for a given n-gram model and test data.
    """
    tokens = tokenize(test_data)
    ngrams = get_ngrams(ngram_model.n, tokens)
    log_prob_sum = 0
    N = len(tokens)

    for ngram in ngrams:
        context, target_word = ngram
        prob = ngram_model.prob(context, target_word)
        if prob > 0:
            log_prob_sum += math.log2(prob)

    perplexity = 2 ** (-1 / N * log_prob_sum)
    return perplexity

def get_ngrams(n: int, tokens: list) -> list:
    tokens = (n-1)*['<START>']+tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i])
        for i in range(n-1, len(tokens))]
    return l

class NgramModel(object):
    def __init__(self, n):
        self.n = n
        self.context = {}
        self.ngram_counter = {}

    def update(self, sentence: str) -> None:
        n = self.n
        ngrams = get_ngrams(n, tokenize(sentence))
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

    def prob(self, context, token):
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context
        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)

        summ = 0
        for token in sorted(map_to_probs):
            summ += map_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count: int):
        n = self.n
        context_queue = (n - 1) * ['<START>']
        result = []
        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)
    





def create_ngram_model(n, path):
    m = NgramModel(n)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.split('.')
        for sentence in text:
            sentence += '.'
            m.update(sentence)
    return m

def simple_probability(n, path):
    """
    Calculate simple probability based on word frequencies in the training data.
    """
    m = NgramModel(n)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.split('.')
        for sentence in text:
            sentence += '.'
            m.update(sentence)
    return m
    
def main():
    st.title("Fantasy Lore Generator")

    image = 'fantasy1.jpeg'

    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    # Create a Streamlit navbar
    page = st.sidebar.selectbox("Select a page:", ["Generate Fantasy Lore", "Simple Text Generator"])

    if page == "Generate Fantasy Lore":
        user_input_sentence = st.text_input("Enter the initial sentence:", key="sentence")
        user_input_len_text = st.number_input("Enter how many words are generated:", key="length", step=1, value=10)
        ngram_order = len(user_input_sentence.split()) + user_input_len_text

        if st.button("Generate"):
            m = create_ngram_model(ngram_order, 'data_final.txt')
            generated_text = m.generate_text(user_input_len_text)
            perplexity_score = perplexity(m, user_input_sentence + generated_text)

            st.divider()
            st.markdown('Output:')
            st.success(f'{user_input_sentence} {generated_text}')
            st.text(f'Created with {ngram_order} gram model\nPerplexity Score: {perplexity_score:.2f}')

    elif page == "Simple Text Generator":
        user_input_words = st.text_input("Enter your words (space-separated):", key="input_words")
        user_input_len_text = st.number_input("Enter how many words to generate:", key="gen_length", step=1, value=10)

        if st.button("Generate"):
            input_words = user_input_words.split()
            if len(input_words) >= 2:
                context = tuple(input_words[-2:])
                
                ngram_order = 3  # trigram
                m = simple_probability(ngram_order, 'data_final.txt')
                generated_text = input_words[:]  # Start with input words

                for _ in range(user_input_len_text):
                    next_word = m.random_token(context)
                    generated_text.append(next_word)
                    if next_word == '.':
                        context = tuple(input_words[-2:])
                    else:
                        context = (context[-1], next_word)

                generated_text = ' '.join(generated_text)

                # Calculate simple probability-based perplexity
                perplexity_score = perplexity(m, generated_text)

                st.divider()
                st.markdown('Output:')
                st.success(generated_text)
                st.text(f'Created with {ngram_order} gram model\nPerplexity Score: {perplexity_score:.2f}')

    else:
        st.error("Please enter at least two words for text generation.")

if __name__ == "__main__":
    main()
