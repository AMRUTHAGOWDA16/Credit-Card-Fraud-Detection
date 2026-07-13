from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import joblib
import os
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this'

# Database setup
def init_db():
    conn = sqlite3.connect('fraud_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    
    # Create default admin user (password: admin123)
    hashed_pw = generate_password_hash('admin123')
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  ('admin', hashed_pw))
    except:
        pass  # User already exists
    
    conn.commit()
    conn.close()

init_db()

# Check if user is logged in
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('fraud_detection.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user'] = username
            return redirect(url_for('menu'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/menu')
@login_required
def menu():
    return render_template('menu.html')

@app.route('/analysis')
@login_required
def analysis():
    # Check if dataset exists
    if not os.path.exists('data/creditcard.csv'):
        flash('Dataset not found! Please upload creditcard.csv to data folder')
        return redirect(url_for('menu'))
    
    # Load dataset for analysis
    df = pd.read_csv('data/creditcard.csv')
    
    # Basic statistics
    total_transactions = len(df)
    fraud_count = df['Class'].sum()
    legitimate_count = total_transactions - fraud_count
    fraud_percentage = (fraud_count / total_transactions) * 100
    
    # Amount statistics
    avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
    avg_legit_amount = df[df['Class'] == 0]['Amount'].mean()
    
    stats = {
        'total_transactions': total_transactions,
        'fraud_count': int(fraud_count),
        'legitimate_count': int(legitimate_count),
        'fraud_percentage': round(fraud_percentage, 2),
        'avg_fraud_amount': round(avg_fraud_amount, 2),
        'avg_legit_amount': round(avg_legit_amount, 2)
    }
    
    return render_template('analysis.html', stats=stats)

@app.route('/model_building')
@login_required
def model_building():
    # Check if models exist
    models_exist = os.path.exists('models/model.joblib') and os.path.exists('models/scaler.joblib')
    
    if models_exist:
        # Load model metrics if they exist
        if os.path.exists('models/metrics.joblib'):
            metrics = joblib.load('models/metrics.joblib')
        else:
            metrics = None
    else:
        metrics = None
    
    return render_template('model_building.html', models_exist=models_exist, metrics=metrics)

@app.route('/model_testing', methods=['GET', 'POST'])
@login_required
def model_testing():
    if request.method == 'POST':
        # Check if model exists
        if not os.path.exists('models/model.joblib'):
            flash('Model not trained yet! Please train the model first.')
            return redirect(url_for('model_building'))
        
        try:
            # Load model and scaler
            model = joblib.load('models/model.joblib')
            scaler = joblib.load('models/scaler.joblib')
            
            # Get form data
            transaction_amount = float(request.form['transaction_amount'])
            transaction_freq = int(request.form['transaction_freq'])
            prev_trans_hist = int(request.form['prev_trans_hist'])
            purchase_amount = float(request.form['purchase_amount'])
            cvv = int(request.form['cvv'])
            billing_verify = int(request.form['billing_verify'])
            time_since_last = float(request.form['time_since_last'])
            avg_trans_amount = float(request.form['avg_trans_amount'])
            failed_attempts = int(request.form['failed_attempts'])
            unusual_pattern = int(request.form['unusual_pattern'])
            
            # Create feature array (28 V features + Time + Amount)
            # For simplicity, we'll use the 10 features provided and pad with zeros
            features = np.zeros(30)
            features[0] = time_since_last  # Time
            features[1] = transaction_freq
            features[2] = prev_trans_hist
            features[3] = purchase_amount
            features[4] = cvv
            features[5] = billing_verify
            features[6] = avg_trans_amount
            features[7] = failed_attempts
            features[8] = unusual_pattern
            features[29] = transaction_amount  # Amount
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
                'fraud_probability': round(probability[1] * 100, 2),
                'legit_probability': round(probability[0] * 100, 2)
            }
            
            return render_template('model_testing.html', result=result)
            
        except Exception as e:
            flash(f'Error during prediction: {str(e)}')
            return render_template('model_testing.html')
    
    return render_template('model_testing.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)