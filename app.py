import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from kmodes.kprototypes import KPrototypes
import joblib

app = Flask(__name__)

# Load the initial patient data
patients = pd.read_csv('LLS_patient_data (2).csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cluster_links')
def cluster_links():
    return render_template('links.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    global patients

    # Get the user inputs from the form
    name = request.form['name']
    relationship = request.form['relationship']
    gender = request.form['gender']
    age = int(request.form['age'])
    ethnicity = request.form['ethnicity']
    diagnosis_by_location = request.form['diagnosis_by_location']
    cancer_diagnosis = request.form['cancer_diagnosis']
    cancer_stage = int(request.form['cancer_stage'])
    time_with_cancer = int(request.form['time_with_cancer'])
    primary_language = request.form['primary_language']
    parenting_situation = request.form['parenting_situation']
    personal_concern = request.form['personal_concern']
    phone_number = request.form['phone_number']
    email = request.form['email']

    # Create a new data entry
    new_entry = {
        'Full Name': name,
        'Relationship': relationship,
        'Gender': gender,
        'Age': age,
        'Ethnicity': ethnicity,
        'Diagnosis by Location': diagnosis_by_location,
        'Cancer Diagnosis': cancer_diagnosis,
        'Cancer Stage': cancer_stage,
        'Time with Cancer (months)': time_with_cancer,
        'Primary Speaking Language': primary_language,
        'Parenting Situation': parenting_situation,
        'Personal Concern': personal_concern,
        'Phone Number': phone_number,
        'Email': email
    }

    print ("Initial State:")
    print(f"Number of patients: {len(patients)}")
    # Append the new entry to the patient dataframe
    patients = patients.append(new_entry, ignore_index=True)

    # Perform clustering
    categorical_cols = ['Relationship', 'Gender', 'Ethnicity', 'Diagnosis by Location', 'Cancer Diagnosis', 'Primary Speaking Language', 'Parenting Situation', 'Personal Concern']
    numerical_cols = ['Age', 'Cancer Stage', 'Time with Cancer (months)']
    
    patients_for_clustering_numerical = patients[numerical_cols].values
    patients_for_clustering_categorical = patients[categorical_cols].values
    patients_for_clustering = np.hstack((patients_for_clustering_numerical, patients_for_clustering_categorical))
    
    kproto = KPrototypes(n_clusters=8, init='Cao', n_init=10, random_state=123)
    clusters = kproto.fit_predict(patients_for_clustering, categorical=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))))
    print("After Clustering:")
    print(f"Clusters = {clusters}")
    print(f"Number of clusters = {len(set(clusters))}")
    new_data_index = len(patients) - 1
    predicted_cluster = clusters[new_data_index]
    
    new_order = ['Full Name', 'Relationship', 'Gender','Age', 'Ethnicity', 'Diagnosis by Location', 'Cancer Diagnosis', 'Cancer Stage', 'Time with Cancer (months)', 'Primary Speaking Language', 'Parenting Situation', 'Personal Concern', 'Phone Number', 'Email']
    patients = patients.reindex(columns=new_order)
    filtered_patients = patients[clusters == predicted_cluster]

    # Save the kproto model to a file
    joblib.dump(kproto, 'kproto_model.pkl')

    return render_template('results.html', cluster_data=filtered_patients.to_dict(orient='records'), cluster=predicted_cluster)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
