# Clothing Review Recommender (Thread & Form)

A Flask website for browsing a clothing catalogue and writing reviews. When a customer writes a review, a trained machine learning model reads it and **suggests whether they would recommend the item**. The reviewer sees the suggestion and has the final say before the review is published.

Built end to end: text preprocessing, a trained classifier, a web app, a database and automated tests.

## Features

- **Browse** 1,095 clothing items, 12 per page, with category filters (Dresses, Knits, Blouses and more)
- **Search** that treats word forms as equivalent, so "dress" and "dresses" return the same items
- **Item pages** showing the description, category and all customer reviews
- **Write a review** with a title, text and 1 to 5 rating
- **Model suggestion**: a logistic regression classifier predicts "recommended" or "not recommended" from the review text, and the reviewer can keep or override it
- **Persistent reviews**: confirmed reviews are saved in SQLite, get their own permanent URL and survive restarts

## The model

- **Algorithm:** logistic regression on word-count vectors (scikit-learn)
- **Input:** review title and review text combined. The star rating is deliberately not used
- **Preprocessing:** lowercasing, a fixed token pattern, stopword removal for titles, and a fixed vocabulary for review text
- **Performance on held-out data (20% of reviews):** about 0.89 accuracy and 0.81 macro F1
- **Training and prediction use the same code path** (`nlp.py`), and a test checks that the website produces exactly the same tokens as the training data, which prevents a common source of silent bugs

The model can be wrong, which is why the human gets the final decision. If a review contains no words the model knows, the site asks for a more detailed description instead of guessing.

## Engineering details

- **Search normalisation:** simple rules reduce words to root forms (for example "dresses" to "dress" and "jeans" to "jean") without downloading extra language tools. An item matches when every search word appears in its title, description or category
- **Messy data handling:** 195 clothing IDs had conflicting titles or descriptions across rows, so the app shows the version that appears most often for each ID
- **Two-step review flow:** the preview step produces a signed, time-limited token (`itsdangerous`, 1 hour expiry), so the confirm step cannot be forged or tampered with
- **Duplicate-safe saving:** each draft has a unique nonce, so submitting the confirm form twice saves only one review
- **Security basics:** CSRF tokens on every form, input validation on all fields, request size limit, and HTML escaping in templates
- **Automated tests:** five integration tests cover every item page, search behaviour, the full review flow with prediction override, persistence across app restarts, and validation and security checks

## Tech stack

Python, Flask, scikit-learn, SQLite, Jinja2, HTML and CSS, pandas, NumPy, SciPy, joblib

## Project structure

```
app.py                Flask app: pages, search, review workflow, database
nlp.py                Text preprocessing shared by training and prediction
models/               Saved classifier (recommender.joblib)
data/                 Vocabulary and stopword list, plus the dataset (see below)
templates/            HTML templates
static/               CSS
tests/                Integration tests
requirements.txt      Python dependencies
```

## Running it locally

Requires Python 3.11 or newer.

**Windows**
```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

**Mac / Linux**
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Then open http://127.0.0.1:5000. The saved model is included, so nothing needs to be trained. The `instance/` folder where new reviews are stored is created automatically.

To run the tests:
```
python -m unittest discover -s tests -v
```

## About the data

The app expects a file at `data/assignment3_II.csv` with these columns: `Clothing ID`, `Clothes Title`, `Clothes Description`, `Division Name`, `Department Name`, `Class Name`, `Title`, `Review Text`, `Rating` and `Recommended IND`.

This is a modified version of the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset on Kaggle (19,662 reviews). The product titles and descriptions were added as artificial metadata, so they are not real product data. The modified file is not included in this repository.

## Limitations

- The recommendation is a simple bag-of-words model, so it does not understand word order, sarcasm or context
- There are no real product photos in the data, so the site uses simple illustrated clothing graphics
- Student project: there are no user accounts, purchases or deployment. It runs locally only

## What I learned

- Keeping training and prediction preprocessing identical, and testing that they match
- Designing a human-in-the-loop workflow around an imperfect model
- Handling messy real-world data, such as conflicting metadata for the same product
- Building a secure, persistent web workflow with signed tokens, CSRF protection and idempotent saves

## Sources

- Original dataset: Women's E-Commerce Clothing Reviews (Kaggle), modified for a university course
- Built with Flask, scikit-learn, pandas, NumPy, SciPy and joblib
- AI assistance was used while building the project and writing this README

---

*Author: [Your Name] | [LinkedIn URL] | [Email]*
