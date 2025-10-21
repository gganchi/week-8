from collections import defaultdict
import numpy as np


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):
        """
        Builds a dictionary where each key is a token in the corpus,
        and each value is a list of tokens that follow that key.
        """
        # initialize as defaultdict(list) for automatic list creation
        term_dict = defaultdict(list)

        # iterate through corpus (excluding last token)
        for i in range(len(self.corpus) - 1):
            current_token = self.corpus[i]
            next_token = self.corpus[i + 1]
            term_dict[current_token].append(next_token)

        # store dictionary in the object
        self.term_dict = dict(term_dict)
        return self.term_dict


    def generate(self, seed_term=None, term_count=15, random_state=None):
        """
        Generate text using the Markov property.
        """
        # Ensure the transition dictionary exists
        if self.term_dict is None:
            self.get_term_dict()

        rng = np.random.default_rng(random_state)

        # If the dictionary is empty, return nothing
        if not self.term_dict:
            return ""

        keys = list(self.term_dict.keys())

        # Choose or validate the starting token
        if seed_term is None:
            current = rng.choice(keys)
        else:
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus.")
            current = seed_term

        output = [current]

        # Generate subsequent terms
        for _ in range(max(0, term_count - 1)):
            followers = self.term_dict.get(current, [])

            if not followers:
                # If current word has no followers, restart randomly
                current = rng.choice(keys)
                output.append(current)
                continue

            # Choose next word at random
            current = rng.choice(followers)
            output.append(current)

        return " ".join(output)