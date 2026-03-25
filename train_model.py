import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

print("⏳ Loading dataset...")

# Define the standard column names for the Kaggle Fake Job Prediction dataset
COL_NAMES = [
    'job_id', 'title', 'location', 'department', 'salary_range', 
    'company_profile', 'description', 'requirements', 'benefits', 
    'telecommuting', 'has_company_logo', 'has_questions', 'employment_type', 
    'required_experience', 'required_education', 'industry', 'function', 'fraudulent'
]

try:
    # 1. LOAD DATA
    # We load with header=None so the first row is treated as data, not titles
    # We verify if 'title' exists, if not we reload with manual names
    df = pd.read_csv('fake_job_postings.csv')
    
    if 'title' not in df.columns:
        print("⚠️ Header missing. Reloading with manual column names...")
        df = pd.read_csv('fake_job_postings.csv', header=None, names=COL_NAMES)

    print(f"✅ Dataset loaded: {len(df)} jobs.")

except FileNotFoundError:
    print("❌ Error: 'fake_job_postings.csv' not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# 2. CLEAN DATA
df.fillna('', inplace=True)

# Combine relevant columns
try:
    df['text'] = df['title'] + " " + df['description'] + " " + df['requirements']
except KeyError as e:
    print(f"❌ Column Error: {e}") 
    print("Available columns:", df.columns.tolist())
    exit()

X = df['text']
y = df['fraudulent']

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. BUILD PIPELINE
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# 5. TRAIN
print("⚙️  Training model...")
pipeline.fit(X_train, y_train)

# 6. EVALUATE
print("✅ Training Complete.")
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))

# 7. SAVE
joblib.dump(pipeline, 'fraud_job_model.pkl')
print("💾 Model saved as 'fraud_job_model.pkl'")
