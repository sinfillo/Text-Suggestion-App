# text_suggestion_app/my_models.py
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Union
import re, math

TOKEN_RE = re.compile(r"""
    [A-Za-z]+(?:[-'][A-Za-z]+)*   # слова, также включаем слова с дефисами и апострофами
  | \d+(?:[./:-]\d+)*             # простые числа + даты + время
  | [.,!?;:()"\[\]]               # пунктуация
""", re.VERBOSE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

PUNCT = {".", ",", "!", "?", ";", ":", ")", "(", '"', "'"}
CLOSE_ONLY = {")"}

def _bad_token(token: str) -> bool:
    if token in CLOSE_ONLY:
        return True
    return not re.match(r"^[A-Za-zА-Яа-яЁё0-9][\w'\-]*$|^[.,!?;:()]$", token) # знак препинания или слово, которое начинается с буквы или цифры

def _punct_penalty(token: str) -> float:
    return 0.5 if token in PUNCT else 1.0

def _score(path_sz, log_prob, alpha=0.7) -> float:
    return log_prob / (max(1, path_sz) ** alpha)

class PrefixTreeNode:
    def __init__(self):
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False

class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()
        for token in vocabulary:
            self._insert(token)
    
    def _insert(self, token: str):
        node = self.root
        for sym in token:
            next = node.children.get(sym)
            if next is None:
                next = PrefixTreeNode()
                node.children[sym] = next
            node = next
        node.is_end_of_word = True
    
    def _find_node(self, prefix: str):
        node = self.root
        for sym in prefix:
            node = node.children.get(sym)
            if node is None:
                return None
        return node

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """
        out = []
        i = self._find_node(prefix)
        if i is None:
            return out

        st = [(i, prefix)]

        while len(st) > 0:
            node, path = st.pop()
            if node.is_end_of_word:
                out.append(path)
            for sym, child in sorted(node.children.items(), reverse=True):
                st.append((child, path + sym))
        return out

class WordCompletor:
    def __init__(self, corpus, min_count: int = 1):
        """
        corpus: list – корпус текстов
        """
        tokens_cnt = Counter()
        for tokens in corpus:
            if tokens:
                tokens_cnt.update(tokens)

        self.tokens_cnt = {word: cnt for word, cnt in tokens_cnt.items() if cnt >= min_count} # если хотим почистить самые редкие            
        self.vocab = list(self.tokens_cnt.keys())
        self.total = sum(self.tokens_cnt.values())

        self.prefix_tree = PrefixTree(self.vocab)

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words, probs = [], []
        words_by_prefix = [word for word in self.prefix_tree.search_prefix(prefix) if not _bad_token(word)]
        if not words_by_prefix:
            return [], []
        
        words_probs = []
        for word in words_by_prefix:
            words_probs.append((word, self.tokens_cnt.get(word, 0) / self.total))
        words_probs.sort(key=lambda x: (-x[1], x[0]))

        for word, prob in words_probs:
            words.append(word)
            probs.append(prob)
        
        return words, probs

class NGramLanguageModel:
    def __init__(self, corpus, n):
        self.n = n
        # self.counts = {}
        # self.history_total = {}
        self.counts = defaultdict(Counter)
        self.history_total = defaultdict(int)
        self.total_tokens = 0

        for tokens in corpus:
            if len(tokens) == 0:
                continue
            cnt_tokens = len(tokens)
            if self.n == 0:
                self.counts[()].update(tokens)
                self.history_total[()] += cnt_tokens
                self.total_tokens += cnt_tokens
                continue

            k = self.n
            if cnt_tokens <= k:
                continue
            for i in range(k, cnt_tokens):
                history = tuple(tokens[i-k:i])
                self.counts[history][tokens[i]] += 1
                self.history_total[history] += 1

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """
        next_words, probs = [], []
        if self.n == 0:
            history = ()
        else:
            if len(prefix) == 0 or len(prefix) < self.n:
                return [], []
            history = tuple(prefix[-self.n:])

        counter = self.counts.get(history)
        if not counter:
            return [], []

        total = self.total_tokens
        if self.n > 0:
            total = self.history_total[history]

        words_probs = []
        for word, cnt in counter.items():
            words_probs.append((word, cnt / total))
        words_probs.sort(key=lambda x: (-x[1], x[0]))

        for word, prob in words_probs:
            next_words.append(word)
            probs.append(prob)

        return next_words, probs

class TextSuggestionOpt:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model
    
    def _ensure_tokens(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            return tokenize(text) 
        return list(text)

    def suggest_text(self, text: Union[str, list], n_words=3, n_texts=1, top_k=1) -> list[list[str]]:

        suggestions = []
        cand_words, cand_probs = [], []
        tokens = self._ensure_tokens(text)
        if not tokens:
            return suggestions

        prefix = tokens[-1]
        cand_words, cand_probs = self.word_completor.get_words_and_probs(prefix)
        if not cand_words:
            cand_words.append(prefix)
            cand_probs.append(1.0)

        words_probs = list(zip(cand_words, cand_probs))
        bext_k_words_probs_init = sorted(words_probs, key=lambda word_prob: word_prob[1], reverse=True)[:top_k]

        beams = []
        for word_init, prob_init in bext_k_words_probs_init:
            history_init = tokens[:-1] + [word_init]
            beams.append((history_init, [word_init], math.log(prob_init)))

        if n_words == 0:
            beams.sort(key=lambda t: t[2], reverse=True)
            for _, path, _ in beams[:n_texts]:
                suggestions.append(path)
            return suggestions

        for _ in range(n_words):
            next_beams = []
            for history, path, log_prob in beams:
                next_words, next_probs = self.n_gram_model.get_next_words_and_probs(history)
                next_words_probs = list(zip(next_words, next_probs))
                # print("path: ", path)
                # print("history: ", history)
                # print("probs: ", words_probs)
                # print("---------------------------")
                if not next_words:
                    next_beams.append((history, path, log_prob))
                    continue

                next_words_probs_clear = []
                for word, prob in next_words_probs:
                    if not _bad_token(word):
                        next_words_probs_clear.append((word, prob))
                bext_k_words_probs = sorted(next_words_probs_clear, key=lambda word_prob: word_prob[1], reverse=True)[:top_k]
                for word, prob in  bext_k_words_probs:
                    pen = _punct_penalty(word)
                    new_history = history + [word]
                    new_path = path + [word]
                    next_beams.append((new_history, new_path, log_prob + math.log(prob * pen + 1e-12)))

            next_beams.sort(key=lambda text: _score(len(text[1]), text[2], alpha=0.7), reverse=True)
            beams = next_beams[:n_texts]

        for _, path, _ in beams:
            suggestions.append(path)
        return suggestions
