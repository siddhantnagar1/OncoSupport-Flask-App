from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory
import pandas as pd
from kmodes.kprototypes import KPrototypes
import joblib
import os
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

# Load the initial patient data
patients = pd.read_csv('oncosupport_dataset.csv')

# Simplified user storage
users = {}

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists')
        else:
            users[username] = password
            session['user'] = username
            flash('Account created successfully')
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('signup'))

user_storage_file = 'users.csv'
if not os.path.exists(user_storage_file):
    with open(user_storage_file, 'w') as f:
        f.write('username,password\n')

def load_users():
    return pd.read_csv(user_storage_file)

def save_user(username, password):
    with open(user_storage_file, 'a') as f:
        f.write(f'{username},{password}\n')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Extract form data
        form_data = {
            'Full Name': request.form['name'],
            'Gender': request.form['gender'],
            'Age': int(request.form['age']),
            'Ethnicity': request.form['ethnicity'],
            'Diagnosis by Location': request.form['diagnosis_by_location'],
            'Cancer Diagnosis': request.form['cancer_diagnosis'],
            'Cancer Stage': int(request.form['cancer_stage']),
            'Time with Cancer (months)': int(request.form['time_with_cancer']),
            'Primary Speaking Language': request.form['primary_language'],
            'Parenting Situation': request.form['parenting_situation'],
            'Personal Concern': request.form['personal_concern'],
            'Hospital': request.form['hospital'],
            'Phone Number': request.form['phone_number'],
            'Email': request.form['email']
        }

        global patients
        patients = pd.concat([patients, pd.DataFrame([form_data])], ignore_index=True)
        
        # Perform clustering
        hospital_patients = patients[patients['Hospital'] == form_data['Hospital']]
        categorical_cols = [
            'Gender', 
            'Ethnicity', 
            'Diagnosis by Location', 
            'Cancer Diagnosis', 
            'Primary Speaking Language', 
            'Parenting Situation', 
            'Personal Concern'
        ]
        numerical_cols = ['Age', 
                          'Cancer Stage', 
                          'Time with Cancer (months)']

        hospital_patients_combined = hospital_patients[numerical_cols + categorical_cols].values
        kproto = KPrototypes(n_clusters=5, init='Cao', n_init=10, random_state=0)
        clusters = kproto.fit_predict(hospital_patients_combined, categorical=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))))
        hospital_patients['Cluster'] = clusters

        # Save the model (optional, if you need to reuse it)
        model_path = 'kproto_model.pkl'
        joblib.dump(kproto, model_path)

        # Determine the cluster of the new patient
        new_patient_cluster = int(hospital_patients.iloc[-1]['Cluster'])
        cluster_data = hospital_patients[hospital_patients['Cluster'] == new_patient_cluster].to_dict(orient='records')

        # Store results in session
        session['cluster_data'] = cluster_data
        session['cluster'] = new_patient_cluster
        session['hospital'] = form_data['Hospital']
        
        return redirect(url_for('results'))
    
    return render_template('index.html')

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Retrieve user's cluster results from session
    cluster_data = session.get('cluster_data', [])
    cluster = session.get('cluster', None)
    hospital = session.get('hospital', None)
    
    return render_template('results.html', cluster_data=cluster_data, cluster=cluster, hospital=hospital)

@app.route('/download')
def download():
    return send_from_directory(directory='/Users/siddhantnagar/Downloads/OncoSupport Phase I Materials', path='Support Group Rules and Guidelines.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)