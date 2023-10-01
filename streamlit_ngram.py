import streamlit as st
import string
import random
from typing import List

# ideally we would use some smart text tokenizer, but for simplicity use this one


import string
from typing import List

def tokenize(text: str) -> List[str]:
    """
    :param text: Takes input sentence
    :return: tokenized sentence
    """

    text = text.replace("’", " ").replace("‘", " ")

    # Ganti karakter “ dan ” menjadi spasi
    text = text.replace("“", " ").replace("”", " ")
    
    # Menghapus tanda baca dengan mengganti mereka dengan spasi
    for punct in string.punctuation:
        text = text.replace(punct, ' ')

    t = text.lower().split()  # Konversi ke huruf kecil dan split menjadi token
    return t


def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams

    ngrams of tuple form: ((previous wordS!), target word)
    """
    # tokens.append('<END>')
    tokens = (n-1)*['<START>']+tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i])
        for i in range(n-1, len(tokens))]
    return l


class NgramModel(object):

    def __init__(self, n):
        self.n = n

        # dictionary that keeps list of candidate words given context
        self.context = {}

        # keeps track of how many times ngram has appeared in the text before
        self.ngram_counter = {}

    def update(self, sentence: str) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
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
        """
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
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
        """
        :param token_count: number of words to be produced
        :return: generated text
        """
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
            # add back the fullstop
            sentence += '.'
            m.update(sentence)
    return m


def main():
    st.title("Ngram Text Generator")

    # user_input_ngram = st.number_input("Insert the number of n-grams :", key="ngram", step=1, value=3)

    st.divider()

    m = create_ngram_model(6, 'data_final.txt')

    user_input_sentence = st.text_input(
        "Enter the initial sentence :", key="sentence")

    user_input_len_text = st.number_input(
        "Enter how many words are generated :", key="length", step=1, value=25)

    if st.button("Generate"):
        generated_text = m.generate_text(user_input_len_text)

        st.divider()

        st.markdown('Output :')

        st.success(f'{user_input_sentence} {generated_text}')


if __name__ == "__main__":
    main()
