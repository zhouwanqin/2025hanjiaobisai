import json
import os
import uuid
from datetime import datetime

DATA_DIR = "data"
ESSAYS_FILE = os.path.join(DATA_DIR, "essays.json")
REVIEWS_FILE = os.path.join(DATA_DIR, "reviews.json")
DISCUSSIONS_FILE = os.path.join(DATA_DIR, "discussions.json")

os.makedirs(DATA_DIR, exist_ok=True)

def init_file(file_path, default=[]):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)

def load_json(file_path, default=[]):
    init_file(file_path, default)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return default

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_essays():
    return load_json(ESSAYS_FILE)

def save_essay(essay):
    essays = load_essays()
    essay['id'] = essay.get('id') or str(uuid.uuid4())
    essay['created_at'] = essay.get('created_at') or datetime.now().strftime("%Y-%m-%d %H:%M")
    essays.append(essay)
    save_json(ESSAYS_FILE, essays)
    return essay['id']

def load_reviews(essay_id):
    reviews = load_json(REVIEWS_FILE)
    return [r for r in reviews if r["essay_id"] == essay_id]

def save_review(review):
    reviews = load_json(REVIEWS_FILE)
    review['id'] = str(uuid.uuid4())
    review['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    reviews.append(review)
    save_json(REVIEWS_FILE, reviews)

def load_discussions(essay_id):
    discussions = load_json(DISCUSSIONS_FILE)
    return [d for d in discussions if d["essay_id"] == essay_id]

def add_discussion_message(message):
    discussions = load_json(DISCUSSIONS_FILE)
    message['id'] = str(uuid.uuid4())
    message['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    discussions.append(message)
    save_json(DISCUSSIONS_FILE, discussions)
