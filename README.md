# Fake News & Misinformation Detector

A machine learning-based application that analyses news articles and helps users investigate potentially misleading information through evidence-based research.

## Features

- **Machine Learning Prediction:** Classifies news text as Fake-labelled or Real-labelled and displays probability estimates.
- **Evidence Search:** Searches for related information and supporting evidence.
- **Live News Coverage:** Retrieves related news articles from Google News.
- **Wikipedia Background:** Provides background information about relevant topics.
- **Fact-Checking Resources:** Provides links to Google Fact Check Explorer and other research sources.
- **NLI Evidence Assessment:** Uses Natural Language Inference (NLI) to assess the relationship between a claim and available evidence.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy
- ONNX Runtime
- Tokenizers
- BeautifulSoup
- Requests

## Requirements

- Python 3.12
- Internet connection

## Installation and Usage

1. Download or clone this repository.
2. Open the project folder in a terminal.
3. Install the required dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Important Note

This application is an experimental misinformation research tool. Machine learning predictions and NLI assessments are not definitive proof that a news article or claim is true or false.

Users should independently examine the original sources and available evidence before drawing conclusions.

## Project Status

Developed as an academic project.
