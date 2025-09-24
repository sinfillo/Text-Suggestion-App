# Text-Suggestion-App
Reflex web app for smart text suggestions using N-gram language models

## Installation & Setup

**Clone the repository**

```bash
git clone git@github.com:sinfillo/Text-Suggestion-App.git
cd Text-Suggestion-App
```

**Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Prepare tokens**
- Place your raw dataset `emails.csv` into the project root.
- Generate tokens:
    ```bash
    python make_tokens.py --input emails.csv --output data/emails_tokens.pkl
    ```
- This will create `data/emails_tokens.pkl`

**Build the suggestion text model**
```bash
python build_sugg_model.py --tokens-pkl data/emails_tokens.pkl --min-count 3 --n 2 --out data/sugg_model.pkl
```
This will create `data/sugg_model.pkl`

**Run the app**
```bash
reflex run
```
Open `http://localhost:3000`

**Help & CLI options**

Both helper scripts come with built-in CLI help:
- For **tokenization**:
    ```bash
    python make_tokens.py --help
    ```
- For **model building**:
    ```bash
    python build_sugg_model.py --help
    ```