
# 🛡️ RIO
## Intelligent Cyberbullying Detection Framework

RIO is an AI-powered cyberbullying detection framework designed to analyse text comments and identify potentially harmful content using a fine-tuned DeBERTa-v3 transformer model.

## 🚀 Features

- AI-based text toxicity analysis
- Six toxicity categories
- DeBERTa-v3 transformer integration
- Interactive Streamlit dashboard
- SQLite database integration
- Automatic analysis history storage
- Confidence score display

## 🧠 Detection Categories

1. Toxic
2. Severe Toxic
3. Obscene
4. Threat
5. Insult
6. Identity Hate

## 🛠️ Technologies Used

- Python
- DeBERTa-v3
- PyTorch
- Hugging Face Transformers
- Streamlit
- SQLite
- Scikit-learn

## 📁 Project Structure

```text
RIO/
├── app.py
├── database.py
├── requirements.txt
├── README.md
├── rio_database.db
├── models/
│   └── deberta_v3/
└── dataset/
```

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project folder:

```bash
cd RIO
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 💾 Database

RIO uses SQLite to store:

- Submitted comments
- Predicted toxicity categories
- Confidence scores
- Analysis timestamps

## ⚠️ Disclaimer

RIO is an academic prototype. Its predictions may be inaccurate and should not be treated as definitive judgments. Human review is recommended.

## 👩‍💻 Project Status

Development in progress.
