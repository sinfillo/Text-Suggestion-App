import re

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
NO_SPACE_AFTER  = {"(", "[", "{"}

def format_tokens(tokens: list[str]):
    '''
    Склеивает строку из suggestа по правилам.
    Не ставит пробел перед знаками из NO_SPACE_BEFORE
    Не ставит пробел после символов из NO_SPACE_AFTER
    '''
    out = ""
    for token in tokens:
        if out == "":
            out = token
            continue
        if token in NO_SPACE_BEFORE:
            out = out.rstrip() + token
        elif out[-1] in NO_SPACE_AFTER:
            out += token
        else:
            out += " " + token
    return out

def _last_token(text: str):
    '''
    Поиск последнего непробельного фрагмента текста перед группой пробелами
    '''
    match = re.search(r"\S+(?=\s*$)", text)
    if match:
        return match.group(0)
    return None

def _get_style(word: str):
    '''
    Стиль регистра слова
    '''
    if word == "":
        return "lower"
    if word.isupper():
        return "upper"
    if word[0].isupper() and word[1:].islower():
        return "title"
    return "lower"

def _need_upper_sym(prev_text: str) -> bool:
    '''
    Нужно ли начинать слово с заглавной буквы -- строка заканчивается на ".!?"
    '''
    return bool(re.search(r"[.!?]\s*$", prev_text))

def _detect_style(text: str):
    last = _last_token(text)
    if _need_upper_sym(text):
        return "title"
    return _get_style(last or "")

def _apply_style(word: str, style: str) -> str:
    if style == "upper":
        return word.upper()
    if style == "title":
        return word.capitalize()
    return word

def _smart_append(base_text: str, tokens: list[str]) -> str:
    '''
    Дописывание токенов в конец набранного текста
    '''
    clear_text = base_text.rstrip()
    for token in tokens:
        if clear_text == "":
            clear_text = token
            continue
        if token in NO_SPACE_BEFORE:
            clear_text = re.sub(r"\s+$", "", token) + token # нужно удалить ненужные пробелы у текущего текста
        elif clear_text[-1] in NO_SPACE_AFTER:
            clear_text += token
        else:
            clear_text += " " + token
    return clear_text

def _clamp(cur_val: int, min_val: int, max_val: int):
    return max(min_val, min(max_val, cur_val))