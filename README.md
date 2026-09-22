# Automated Answer Evaluation System

This project implements an automated answer script evaluation system that integrates Natural Language Processing (NLP) and reinforcement learning techniques. The system is designed to extract text from scanned answer scripts, analyze the content, predict scores, and provide a web-based interface for evaluation.

## Features

- Optical Character Recognition (OCR) for extracting text from scanned documents.
- NLP-based feature extraction from answer script content.
- Reinforcement learning techniques for score prediction.
- User-friendly web interface for evaluation and results display.

## Prerequisites

- Python 3.8 or higher

## Project Structure

```
automated-answer-evaluation
├── src
│   ├── text_extraction
│   │   └── extract_text.py
│   ├── feature_extraction
│   │   └── extract_features.py
│   ├── score_prediction
│   │   └── predict_score.py
│   ├── web_evaluation
│   │   ├── app.py
│   │   └── templates
│   │       └── index.html
│   └── utils
│       └── helpers.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd automated-answer-evaluation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install in development mode:
   ```
   pip install -e .
   ```

## Usage

1. Start the web application:
   ```
   python src/web_evaluation/app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000` to access the evaluation system.

## Objectives

- To develop a robust system for automated evaluation of answer scripts.
- To utilize Optical Character Recognition (OCR) for text extraction from scanned documents.
- To implement NLP techniques for feature extraction and score prediction.
- To provide a user-friendly web interface for interaction and results display.

## Testing

Run any existing tests with:
```
python -m pytest
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

This project leverages various libraries and frameworks for OCR, NLP, and web development. Please refer to the `requirements.txt` for a complete list of dependencies.
