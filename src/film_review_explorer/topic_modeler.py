import spacy
import pandas as pd
from typing import Iterable, Generator, List
from spacy.tokens import Doc, Token

class TextProcessor:
    """A class for text processing using spaCy."""

    def __init__(self, model: str):
        """
        Initializes a TextProcessor instance.

        Args:
            model (str): The name of the spaCy model to load.
        """
        self.nlp = spacy.load(model)

    @staticmethod
    def check_token(
        token: Token, punct: bool = True, num: bool = True, stopword: bool = True
    ) -> bool:
        """
        Checks if a token meets specified criteria.

        Args:
            token (Token): The token to check.
            punct (bool): Whether to consider punctuation tokens (default=True).
            num (bool): Whether to consider numerical tokens (default=True).
            stopword (bool): Whether to consider stopword tokens (default=True).

        Returns:
            bool: True if the token meets the criteria, False otherwise.
        """
        if punct:
            punct = token.is_punct
        if num:
            num = token.like_num
        if stopword:
            stopword = token.is_stop
        return not (stopword or punct or num)

    @staticmethod
    def get_entities(doc: Doc) -> List[str]:
        """
        Retrieves the entities from a spaCy document.

        Args:
            doc (Doc): The spaCy document.

        Returns:
            List[str]: The list of entity texts.
        """
        return [ent.text for ent in doc.ents]

    @staticmethod
    def get_sentences(doc: Doc) -> List[str]:
        """
        Retrieves the sentences from a spaCy document.

        Args:
            doc (Doc): The spaCy document.

        Returns:
            List[str]: The list of sentence texts.
        """
        return [sentence.text for sentence in doc.sents]

    def get_tokens(
        self,
        doc: Doc,
        lemma: bool = True,
        punct: bool = True,
        num: bool = True,
        stopword: bool = True,
    ) -> List[str]:
        """
        Retrieves the tokens from a spaCy document.

        Args:
            doc (Doc): The spaCy document.
            lemma (bool): Whether to use lemmas instead of raw texts (default=True).
            punct (bool): Whether to exclude punctuation tokens (default=True).
            num (bool): Whether to exclude numerical tokens (default=True).
            stopword (bool): Whether to exclude stopword tokens (default=True).

        Returns:
            List[str]: The list of token texts.
        """
        if lemma:
            return [
                token.lemma_
                for token in doc
                if self.check_token(token, punct=punct, num=num, stopword=stopword)
            ]
        else:
            return [
                token.text
                for token in doc
                if self.check_token(token, punct=punct, num=num, stopword=stopword)
            ]

    def get_ngrams(
        self,
        doc: Doc,
        n: int,
        sep: str = " ",
        punct: bool = True,
        num: bool = True,
        stopword: bool = True,
    ) -> List[str]:
        """
        Retrieves n-grams from a spaCy document.

        Args:
            doc (Doc): The spaCy document.
            n (int): The size of n-grams to retrieve.
            sep (str): The separator string between tokens (default=" ").
            punct (bool): Whether to exclude punctuation tokens (default=True).
            num (bool): Whether to exclude numerical tokens (default=True).
            stopword (bool): Whether to exclude stopword tokens (default=True).

        Returns:
            List[str]: The list of n-grams.
        """
        ngrams = []
        for sentence in doc.sents:
            for i in range(len(sentence) - n + 1):
                if all(
                    self.check_token(token, punct=punct, num=num, stopword=stopword)
                    for token in sentence[i : i + n]
                ):
                    ngrams.append(sep.join(token.text for token in sentence[i : i + n]))

        return ngrams

    def process_zh(
        self, texts: Iterable[str], disable: Iterable[str] = []
    ) -> Generator:
        """
        Processes a batch of Chinese texts.

        Args:
            texts (Iterable[str]): The texts to process.
            disable (Iterable[str]): Components to disable during processing (default=[]).

        Yields:
            Generator: A generator yielding the processed results in the following order:
                      tokens, entities, bigrams, trigrams, quadgrams.
        """
        for doc in self.nlp.pipe(texts, disable=disable):
            tokens = self.get_tokens(doc, lemma=False)
            entities = self.get_entities(doc)
            bigrams = self.get_ngrams(doc, 2, sep="")
            trigrams = self.get_ngrams(doc, 3, sep="")
            quadgrams = self.get_ngrams(doc, 4, sep="")

            yield tokens, entities, bigrams, trigrams, quadgrams

    def process_en(
        self, texts: Iterable[str], disable: Iterable[str] = []
    ) -> Generator:
        """
        Processes a batch of English texts.

        Args:
            texts (Iterable[str]): The texts to process.
            disable (Iterable[str]): Components to disable during processing (default=[]).

        Yields:
            Generator: A generator yielding the processed results in the following order:
                      tokens, entities, bigrams, trigrams, quadgrams.
        """
        for doc in self.nlp.pipe(texts, disable=disable):
            tokens = self.get_tokens(doc)
            entities = self.get_entities(doc)
            bigrams = self.get_ngrams(doc, 2, sep=" ")
            trigrams = self.get_ngrams(doc, 3, sep=" ")
            quadgrams = self.get_ngrams(doc, 4, sep=" ")

            yield tokens, entities, bigrams, trigrams, quadgrams
