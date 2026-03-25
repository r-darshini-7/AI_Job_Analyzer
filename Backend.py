from flask import Flask, render_template, request, jsonify
import joblib
import mysql.connector
import sys

app = Flask(__name__)

# --- 1. DATABASE CONFIGURATION ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        # ⚠️ UPDATE THIS WITH YOUR REAL MYSQL PASSWORD
        password="D@rshini16r", 
        database="fraud_guard"
    )

# --- 2. LOAD AI MODEL ---
try:
    model = joblib.load('fraud_job_model.pkl')
    print("✅ Model loaded successfully.")
except:
    print("⚠️ Model not found! Run 'train_model.py' first.")
    model = None

# --- 3. RISK RULES ENGINE ---
SUSPICIOUS_KEYWORDS = ['urgent', 'immediate start', 'no experience', 'cash', 'kindly', 'work from home']
BANNED_KEYWORDS = ['western union', 'telegram', 'whatsapp', 'wire transfer', 'check', 'money mule']

def calculate_risk(text, ai_score):
    text = text.lower()
    for word in BANNED_KEYWORDS:
        if word in text: return 0.99
    penalty = 0
    for word in SUSPICIOUS_KEYWORDS:
        if word in text: penalty += 0.15
    return min(ai_score + penalty, 0.99)

# --- 4. ROUTES ---

@app.route('/')
def home():
    return render_template('Frontend.html') 

@app.route('/predict', methods=['POST'])
def predict():
    print("🔍 Analysis Request Received...") # Debug print
    if not model: return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    description = data.get('description', '')
    company = data.get('company', '')
    job_type = data.get('job_type', '')
    full_text = f"{company} {job_type} {description}"

    probabilities = model.predict_proba([full_text])[0]
    raw_fraud_prob = probabilities[1]
    final_fraud_prob = calculate_risk(full_text, raw_fraud_prob)
    
    fake_score = round(final_fraud_prob * 100, 2)
    real_score = round(100 - fake_score, 2)
    verdict = "Fake" if fake_score > 50 else "Real"

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO search_history (company_name, job_type, description, prediction_score, verdict) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (company, job_type, description, fake_score, verdict))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Data saved to MySQL.")
    except Exception as e:
        print(f"❌ DB Error: {e}")

    return jsonify({'real_prob': real_score, 'fake_prob': fake_score})

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 1. Daily Stats
        cursor.execute("SELECT DATE_FORMAT(search_date, '%Y-%m-%d') as date, verdict, COUNT(*) as count FROM search_history WHERE search_date >= CURDATE() - INTERVAL 7 DAY GROUP BY date, verdict ORDER BY date ASC")
        daily = cursor.fetchall()

        # 2. Monthly Stats
        cursor.execute("SELECT DATE_FORMAT(search_date, '%Y-%m') as month, verdict, COUNT(*) as count FROM search_history WHERE search_date >= CURDATE() - INTERVAL 12 MONTH GROUP BY month, verdict ORDER BY month ASC")
        monthly = cursor.fetchall()

        # 3. Yearly Stats
        cursor.execute("SELECT YEAR(search_date) as year, verdict, COUNT(*) as count FROM search_history GROUP BY year, verdict ORDER BY year ASC")
        yearly = cursor.fetchall()

        cursor.close()
        conn.close()
        return jsonify({'daily': daily, 'monthly': monthly, 'yearly': yearly})
    except Exception as e:
        print(f"Analytics Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- 5. START SERVER ---
# This block MUST be at the very bottom, NOT indented.
if __name__ == '__main__':
    print("🚀 Starting Flask Server... Please wait...")
    app.run(debug=True)