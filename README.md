# WasteWise Flask Application

## Folder Structure
Place your existing Drive folders (`Models/`, `Database/`, `Dataset/`) directly into the project root alongside `app.py`:

```text
wastewise_flask_app/
├── app.py                  # Flask routes (Home, Waste Analyzer, Ask Me, About)
├── requirements.txt        # Python dependencies
├── .env.example            # Copy to .env and add your GEMINI_API_KEY
├── src/
│   ├── __init__.py
│   ├── wasteloop.py        # Unchanged classification, extraction, feedback & SQLite logic
│   └── rag.py              # Unchanged Gemini + LlamaIndex + ChromaDB regulatory Q&A logic
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── analyzer.html
│   ├── ask.html
│   └── about.html
├── static/
│   └── style.css
├── Models/                 # Copy your existing Models folder here
│   ├── sentence_transformer/
│   └── sentence_classifier.pkl
├── Database/               # Copy your existing Database folder here
│   ├── chromadb/
│   └── wasteloop_mvp.db
└── Dataset/                # Copy your existing Dataset folder here
    └── approved_feedback_dataset.csv
```

## Setup & Run
1. Copy your `Models`, `Database`, and `Dataset` folders from Google Drive into `wastewise_flask_app/`.
2. Create a `.env` file (or copy `.env.example` to `.env`) and set `GEMINI_API_KEY`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask server:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser.
