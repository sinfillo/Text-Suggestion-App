import pickle, pathlib, argparse
from text_suggestion_app.models import WordCompletor, NGramLanguageModel, TextSuggestionOpt

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--tokens-pkl", default="data/emails_tokens.pkl", help="Path to a pickle file with tokenized texts")
    argument_parser.add_argument("--min-count", type=int, default=3, help="Minimum frequency threshold for a token to be included in the vocabulary")
    argument_parser.add_argument("--n", type=int, default=2, help="Length of the context (number of previous tokens used)")
    argument_parser.add_argument("--out", default="data/sugg_model.pkl", help="Path to save the serialized model (pickle file)")
    args = argument_parser.parse_args()

    file_path = pathlib.Path(args.tokens_pkl)
    assert file_path.exists(), f"No file {file_path}"
    with file_path.open("rb") as f:
        object = pickle.load(f)
    emails_tokens = object["tokens"]    

    def it():
        for tokens in emails_tokens:
            if tokens:
                yield tokens

    word_completor = WordCompletor(it(), min_count=args.min_count)
    n_gram_model = NGramLanguageModel(it(), n=args.n)
    text_suggestion = TextSuggestionOpt(word_completor, n_gram_model)

    output_model_path = pathlib.Path(args.out)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with output_model_path.open("wb") as f:
        pickle.dump({"version": 1, "n": args.n, "min_count": args.min_count, "model": text_suggestion}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"OK: saved {output_model_path}")

if __name__ == "__main__":
    main()
