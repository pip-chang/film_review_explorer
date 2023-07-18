import spacy
import pandas as pd
from typing import Iterable, Generator, List
from spacy.tokens import Doc, Token

df = pd.read_json("data.json")


# load processed dataframes in and prepare for topic modeling

# functions for every step of BERTopic


class TextProcessor:
    def __init__(self, model: str):
        self.nlp = spacy.load(model)

    @staticmethod
    def check_token(
        token: Token, punct: bool = True, num: bool = True, stopword: bool = True
    ) -> bool:
        if punct:
            punct = token.is_punct
        if num:
            num = token.like_num
        if stopword:
            stopword = token.is_stop
        return not (stopword or punct or num)

    @staticmethod
    def get_entities(doc: Doc) -> List[str]:
        return [ent.text for ent in doc.ents]

    @staticmethod
    def get_sentences(doc: Doc) -> List[str]:
        return [sentence.text for sentence in doc.sents]

    def get_tokens(
        self,
        doc: Doc,
        lemma: bool = True,
        punct: bool = True,
        num: bool = True,
        stopword: bool = True,
    ) -> List[str]:
        if lemma:
            return [
                token.lemma_
                for token in doc
                if self.check_token(token, punct=punct, num=num, stopword=stopword)
            ]
        else:
            return [
                token
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
        for doc in self.nlp.pipe(texts, disable=disable):
            tokens = self.get_tokens(doc)
            entities = self.get_entities(doc)
            bigrams = self.get_ngrams(doc, 2, sep=" ")
            trigrams = self.get_ngrams(doc, 3, sep=" ")
            quadgrams = self.get_ngrams(doc, 4, sep=" ")

            yield tokens, entities, bigrams, trigrams, quadgrams