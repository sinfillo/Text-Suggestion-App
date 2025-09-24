import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd

class EmailsCleanerFastPlus:
    def __init__(self, lowercase=True, collapse_whitespaces=True,
        save_newlines=False, drop_quotes=True,
        replies: Optional[List[str]] = None, drop_signatures=False,
        signatures_starts: Optional[List[str]] = None, max_chars = 200000,
    ):
        self.lowercase = lowercase
        self.collapse_whitespaces = collapse_whitespaces # схлопывание пробелов и табуляций
        self.save_newlines = save_newlines
        self.drop_quotes = drop_quotes
        self.drop_signatures = drop_signatures
        self.max_chars = max_chars # чтобы не обрабатывать слишком большие тексты
        self._any_whitespaces = re.compile(r"\s+") 
        if replies is None:
            self.replies = []
        else:
            self.replies = tuple(m.lower() for m in replies)
        
        if signatures_starts is None:
            self.signatures_starts = []
        else:
            self.signatures_starts = tuple(s.lower() for s in signatures_starts)

    @staticmethod
    def _body(full_message: str):
        """
        Получаем основной текст сообщения
        """
        if not full_message:
            return ""
        i = full_message.find("\n\n")
        if i != -1:
            return full_message[i+2:]
        return full_message


    def _strip_text(self, text: str):
        out = []
        for line in text.splitlines():
            lns = line.strip()
            if not lns:
                if self.save_newlines: # если хотим сохранить пустые строки
                    out.append("")
                continue

            low = lns.lower()

            # дропаю цитаты
            if self.drop_quotes and lns.startswith(">"):
                continue

            # если мы встретили фразу, по которой должны оборвать хвост сообщения
            if any(m in low for m in self.replies):
                break

            # удаляем подписи, так как я не хочу чтобы предсказывались имена
            if self.drop_signatures and any(low.startswith(s) for s in self.signatures_starts):
                break

            out.append(line)
        
        if self.save_newlines:
            return "\n".join(out)
        return " ".join(out)

    def _normalize(self, text: str):
        if self.lowercase:
            text = text.lower()
        
        # как раз схлопываем пробелы
        if self.collapse_whitespaces:
            if self.save_newlines:
                lines = []
                for base_line in text.splitlines():
                    lines.append(self._any_whitespaces.sub(" ", base_line).strip())

                out, prev_blank = [], False
                for line in lines:
                    blank = (line == "")
                    if blank and prev_blank:
                        continue
                    out.append(line)
                    prev_blank = blank
                text = "\n".join(out).strip()
            else:
                text = self._any_whitespaces.sub(" ", text).strip()
        return text

    def clean_text(self, message: str):
        if not message:
            return ""
        text = self._body(message)
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        text = self._strip_text(text)
        text = self._normalize(text)
        return text

    def transform(self, messages: Iterable[str]):
        return [self.clean_text(m) for m in messages]

TOKEN_RE = re.compile(r"""
    [A-Za-z]+(?:[-'][A-Za-z]+)*   # слова, также включаем слова с дефисами и апострофами
  | \d+(?:[./:-]\d+)*             # простые числа + даты + время
  | [.,!?;:()"\[\]]               # пунктуация
""", re.VERBOSE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", type=Path, default=Path("emails.csv"), help="Path to CSV with emails")
    argument_parser.add_argument("--text-col", type=str, default="message", help="Name of the column with main text")
    argument_parser.add_argument("--out", type=Path, default=Path("data/emails_tokens.pkl"), help="Where to save pickle file")
    args = argument_parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    assert args.text_col in df.columns, (f"Column '{args.text_col}' not found in CSV. " f"Available columns: {list(df.columns)}")

    cleaner = EmailsCleanerFastPlus(
        lowercase=True,
        collapse_whitespaces=True,
        save_newlines=False,
        drop_quotes=True,
        replies=["-----original message-----", "----- forwarded by"],
        drop_signatures=False,
        signatures_starts=["--", "thanks,", "best,", "regards,", "cheers,"],
        max_chars=200000,
    )
    df["text"] = cleaner.transform(df[args.text_col].astype(str).tolist())
    df["tokens"] = df["text"].apply(tokenize)
    df = df[df["tokens"].apply(lambda toks: len(toks) > 0)].reset_index(drop=True)

    tokens = df["tokens"].tolist()
    with args.out.open("wb") as f:
        pickle.dump({"tokens": tokens}, f)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()