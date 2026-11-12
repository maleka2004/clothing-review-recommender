"""Flask catalogue and two-stage, persistent review workflow."""
import csv
import math
import os
import re
import secrets
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
from flask import Flask, abort, redirect, render_template, request, session, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from nlp import ROOT, features

#Handles common word endings without downloading extra language tools.
def normalise(text):
    irregular = {'dresses': 'dress', 'blouses': 'blouse', 'jeans': 'jean', 'pants': 'pant', 'shorts': 'short', 'trousers': 'trouser'}
    words = re.findall(r'[a-z]+', text.lower())
    return {irregular.get(w, w[:-3] + 'y' if w.endswith('ies') else w[:-1] if w.endswith('s') and not w.endswith(('ss', 'us')) else w) for w in words}

def load_catalogue():
    items = {}
    metadata = {}
    with (ROOT / 'data/assignment3_II.csv').open(encoding='utf-8-sig', newline='') as file:
        for index, row in enumerate(csv.DictReader(file)):
            item_id = int(row['Clothing ID'])
            if item_id not in items:
                items[item_id] = dict(id=item_id, title=row['Clothes Title'], description=row['Clothes Description'], division=row['Division Name'] or 'Other', department=row['Department Name'] or 'Other', category=row['Class Name'] or 'Other', reviews=[])
            metadata.setdefault(item_id, Counter()).update([(row['Clothes Title'], row['Clothes Description'], row['Division Name'] or 'Other', row['Department Name'] or 'Other', row['Class Name'] or 'Other')])
            items[item_id]['reviews'].append(dict(id=f'source-{index}', title=row['Title'], body=row['Review Text'], rating=int(row['Rating']), recommended=int(row['Recommended IND']), source=True))
    for item in items.values():
        chosen = metadata[item['id']].most_common(1)[0][0]
        item.update(zip(('title', 'description', 'division', 'department', 'category'), chosen))
        item['search'] = normalise(' '.join(str(item[k]) for k in ['title', 'description', 'division', 'department', 'category']))
        item['average'] = round(sum(r['rating'] for r in item['reviews']) / len(item['reviews']), 1)
    return items

def create_app(config=None):
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32), DATABASE=str(ROOT / 'instance/reviews.sqlite3'), MAX_CONTENT_LENGTH=64 * 1024)
    if config:
        app.config.update(config)
    Path(app.config['DATABASE']).parent.mkdir(parents=True, exist_ok=True)
    @contextmanager
    def db():
        connection = sqlite3.connect(app.config['DATABASE'])
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()
    with db() as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5), recommended INTEGER NOT NULL CHECK(recommended IN (0,1)), predicted INTEGER NOT NULL, created TEXT NOT NULL, nonce TEXT UNIQUE NOT NULL)')
    catalogue = load_catalogue()
    model = joblib.load(ROOT / 'models/recommender.joblib')
    signer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='review-preview')
    categories = sorted(Counter(i['category'] for i in catalogue.values()).items())
    @app.before_request
    def csrf():
        session.setdefault('csrf', secrets.token_hex(24))
        if request.method == 'POST' and request.form.get('csrf') != session['csrf']:
            abort(400, description='Your form expired. Reload the page and try again.')
    @app.context_processor
    def common():
        return dict(categories=categories)
    def item_or_404(item_id):
        if item_id not in catalogue:
            abort(404)
        return catalogue[item_id]
    @app.get('/')
    def home():
        q = request.args.get('q', '').strip()[:200]
        category = request.args.get('category', '')
        query_tokens = normalise(q)
        matches = [i for i in catalogue.values() if (not category or i['category'] == category) and (not q or (bool(query_tokens) and query_tokens.issubset(i['search'])))]
        matches.sort(key=lambda i: (-len(i['reviews']), i['id']))
        try:
            page = max(1, int(request.args.get('page', 1)))
        except ValueError:
            page = 1
        pages = max(1, math.ceil(len(matches) / 12))
        page = min(page, pages)
        return render_template('home.html', items=matches[(page-1)*12:page*12], total=len(matches), q=q, category=category, page=page, pages=pages)
    @app.get('/items/<int:item_id>')
    def item_detail(item_id):
        item = item_or_404(item_id)
        with db() as connection:
            new = connection.execute('SELECT * FROM reviews WHERE item_id=? ORDER BY id DESC', (item_id,)).fetchall()
        return render_template('item.html', item=item, new_reviews=new)
    @app.route('/items/<int:item_id>/reviews/new', methods=['GET', 'POST'])
    def new_review(item_id):
        item = item_or_404(item_id)
        error = None
        if request.method == 'POST':
            title = request.form.get('title', '').strip()
            body = request.form.get('body', '').strip()
            rating = request.form.get('rating', '')
            if not 1 <= len(title) <= 150 or not 1 <= len(body) <= 5000 or rating not in {'1','2','3','4','5'} or not re.search('[a-zA-Z]', body):
                error = 'Enter a title (1–150 characters), review (1–5,000 characters with letters), and a rating from 1 to 5.'
            else:
                vector = model[0].transform([features(title, body)])
                if vector.nnz == 0:
                    error = 'No familiar words were found. Please describe the clothing and your experience in more detail.'
                else:
                    predicted = int(model[1].predict(vector)[0])
                    draft = dict(item_id=item_id, title=title, body=body, rating=int(rating), predicted=predicted, nonce=secrets.token_hex(24), csrf=session['csrf'])
                    return render_template('confirm.html', item=item, draft=draft, token=signer.dumps(draft))
        return render_template('new.html', item=item, error=error), 400 if error else 200
    @app.post('/reviews/confirm')
    def confirm_review():
        try:
            draft = signer.loads(request.form.get('token', ''), max_age=3600)
        except (BadSignature, SignatureExpired):
            abort(400, description='Review preview expired. Please create the review again.')
        if draft['csrf'] != session['csrf'] or request.form.get('recommended') not in {'0', '1'}:
            abort(400, description='Choose a valid recommendation.')
        item_or_404(draft['item_id'])
        with db() as connection:
            connection.execute('INSERT OR IGNORE INTO reviews(item_id,title,body,rating,recommended,predicted,created,nonce) VALUES (?,?,?,?,?,?,?,?)', (draft['item_id'], draft['title'], draft['body'], draft['rating'], int(request.form['recommended']), draft['predicted'], datetime.now(timezone.utc).isoformat(timespec='seconds'), draft['nonce']))
            review_id = connection.execute('SELECT id FROM reviews WHERE nonce=?', (draft['nonce'],)).fetchone()['id']
        return redirect(url_for('review_detail', review_id=review_id))
    @app.get('/reviews/<int:review_id>')
    def review_detail(review_id):
        with db() as connection:
            review = connection.execute('SELECT * FROM reviews WHERE id=?', (review_id,)).fetchone()
        if review is None:
            abort(404)
        return render_template('review.html', review=review, item=item_or_404(review['item_id']))
    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def error_page(error):
        return render_template('error.html', error=error), error.code
    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000)
