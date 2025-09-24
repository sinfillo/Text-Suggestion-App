"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
from rxconfig import config
import pickle, pathlib
import re
from .models import tokenize, TextSuggestionOpt
from .helpers import _apply_style, format_tokens, _smart_append, _detect_style, _last_token, _clamp

PICKLE_PATH = pathlib.Path("data/sugg_model.pkl")
with PICKLE_PATH.open("rb") as f:
    OBJECT = pickle.load(f)
TEXT_SUGGESTION: TextSuggestionOpt = OBJECT["model"]

class State(rx.State):
    text: str = ""
    suggestions: list[str] = []
    n_words: int = 1
    n_texts: int = 5
    top_k: int = 3   

    N_WORDS_MIN = 1
    N_WORDS_MAX = 5
    N_TEXTS_MIN = 1
    N_TEXTS_MAX = 12
    TOP_K_MIN = 1
    TOP_K_MAX = 20

    def set_n_words(self, value):
        self.n_words = _clamp(value[0], self.N_WORDS_MIN, self.N_WORDS_MAX)
        self.on_change(self.text)

    def set_n_texts(self, value):
        self.n_texts = _clamp(value[0], self.N_TEXTS_MIN, self.N_TEXTS_MAX)
        self.on_change(self.text)

    def set_top_k(self, value):
        self.top_k = _clamp(value[0], self.TOP_K_MIN, self.TOP_K_MAX)
        self.on_change(self.text)

    def on_change(self, value: str):
        self.text = value
        tokens = tokenize(value.lower())
        if len(tokens) == 0:
            self.suggestions = []
            return
        
        variants = TEXT_SUGGESTION.suggest_text(tokens, n_words=self.n_words, n_texts=self.n_texts, top_k=self.top_k)

        style = _detect_style(self.text)

        self.suggestions = []
        for var in variants:
            if len(var) == 0:
                continue
            var_copy = list(var)
            var_copy[0] = _apply_style(var_copy[0], style)
            self.suggestions.append(format_tokens(var_copy))

    def accept(self, var: str):
        words = var.split()
        if len(words) == 0:
            return

        is_last_space = bool(self.text) and self.text[-1].isspace()

        self.suggestions = []

        style = _detect_style(self.text)
        words[0] = _apply_style(words[0], style)

        last = _last_token(self.text)
        if is_last_space:
            if last != "" and words[0].lower() == last.lower(): # чтобы избежать дубликата, когда мы дописываем следующую фразу, а не заменяем
                words = words[1:]
            if len(words) == 0:
                return
            to_append = _smart_append(self.text, words)
            self.text = to_append + " "
        else:
            start_text = re.sub(r"\S+\s*$", "", self.text).rstrip() # убираем недописанное слово и лишние пробелы
            to_append = _smart_append(start_text, [words[0]])
            if len(words) > 1:
                to_append = _smart_append(to_append, words[1:])
            self.text = to_append + " "

        cur_pos = len(self.text)
        return rx.call_script(
            f"""
            const el = document.getElementById('editor');
            if (el) {{ el.focus(); el.setSelectionRange({cur_pos}, {cur_pos}); }}
            """
        )

def labeled_slider(
    label: str,
    value_var,
    on_change,
    min_value: int,
    max_value: int,
    tooltip_text: str,
) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.text(label),
                rx.chakra.tooltip(
                    rx.chakra.badge("i", variant="subtle"),
                    has_arrow=True,
                    label=tooltip_text,
                    placement="top-start"
                ),
                spacing="3",
                align="center",
            ),
            rx.spacer(),
            rx.badge(
                rx.text(value_var),
                color_scheme="blue",
            ),
            align="center",
            width="100%",
        ),
        rx.slider(
            min=min_value,
            max=max_value,
            step=1,
            value=[value_var],
            on_value_commit=on_change,
            width="100%",
        ),
        width="100%",
        spacing="2",
    )


def index() -> rx.Component:
    # Welcome Page (Index)
    controls = rx.card(
        rx.vstack(
            rx.text("Parameters", weight="bold"),
            labeled_slider("n_words", State.n_words, State.set_n_words, State.N_WORDS_MIN, State.N_WORDS_MAX, "How many words to add in one step (1–5)"),
            labeled_slider("n_texts", State.n_texts, State.set_n_texts, State.N_TEXTS_MIN, State.N_TEXTS_MAX, "How many hint options to show (1–12)"),
            labeled_slider("top_k", State.top_k, State.set_top_k, State.TOP_K_MIN, State.TOP_K_MAX, "How many best candidates to consider at each step (1–20)"),
            spacing="3",
            align="center",
            width="100%"
        ),
        size="2",
        style={"width": "800px"}
    )

    return rx.center(
        rx.vstack(
            rx.text_area(
                id="editor",
                value=State.text,
                on_change=State.on_change,
                placeholder="Print...",
                rows="10",
                width="800px"
            ),
            controls,
            rx.flex(
                rx.foreach(
                    State.suggestions,
                    lambda s: rx.button(
                        s,
                        on_click=lambda: State.accept(s),
                        size="2",
                        variant="surface",
                        radius="full"
                    ),
                ),
                gap="4",
                width="800px"
            ),
            spacing="4"
        )
    )


app = rx.App()
app.add_page(index)
