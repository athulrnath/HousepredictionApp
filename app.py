from flask import Flask, request, jsonify, session, redirect, url_for, render_template, flash
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pymongo import MongoClient
import datetime
import bcrypt
from bson import ObjectId
from flask_cors import CORS
import os
import secrets

# Initialize Flask app
app = Flask(__name__)

# CORS Configuration
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})

# Secret key for session management - recommended to use an environment variable for production
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Load the pre-trained CatBoost model
model = joblib.load(r'D:\Athul Project\Model\catboost_model.pkl')

# Assume 'new_dataset' was used for fitting the encoder earlier
new_dataset = pd.read_excel(r'D:\Athul Project\Data\innercity.xlsx')  # Load your dataset
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)

# Convert all categorical columns to strings to ensure uniformity
new_dataset[object_cols] = new_dataset[object_cols].astype(str)

# Recreate the OneHotEncoder and fit on the original dataset's categorical columns
OH_encoder = OneHotEncoder(sparse_output=False, drop='first')
OH_encoder.fit(new_dataset[object_cols])

# Set up MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb+srv://Rithin_1999:Rithin1999@cluster0.pta68bg.mongodb.net/'))
db = client['housepredictions']
user_collection = db['users']
prediction_collection = db['predictions']

# Helper function to check if user is logged in
def is_logged_in():
    return 'user_id' in session

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        email = data.get('email')
        password = data.get('password')

        if not all([first_name, last_name, email, password]):
            return render_template('register.html', error='All fields are required')

        if user_collection.find_one({'email': email}):
            return render_template('register.html', error='Email already exists')

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        user_collection.insert_one({
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.now()
        })

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        email = data.get('email')
        password = data.get('password')

        if not all([email, password]):
            return render_template('login.html', error='Email and password are required')

        user = user_collection.find_one({'email': email})
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return render_template('login.html', error='Invalid email or password')

        session['user_id'] = str(user['_id'])
        return redirect(url_for('predict'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not is_logged_in():
        flash('Unauthorized access. Please log in.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        data = request.form

        # Prepare the data for prediction
        room_bed = float(data.get('room_bed', 0))
        room_bath = float(data.get('room_bath', 0))
        living_measure = float(data.get('living_measure', 0))
        lot_measure = float(data.get('lot_measure', 0))
        quality = float(data.get('quality', 0))
        zipcode = int(data.get('zipcode', 0))
        yr_renovated = int(data.get('yr_renovated', 0))
        basement = int(data.get('basement', 0))
        furnished = int(data.get('furnished', 0))
        yr_built = data.get('yr_built', '2000')
        condition = data.get('condition', '3')
        total_area = data.get('total_area', '1000')
        
        input_data = {
            'dayhours': '20140521T000000',
            'ceil': '2.5',
            'coast': '0',
            'long': '-122.2',
            'yr_built': yr_built,
            'condition': condition,
            'total_area': total_area,
            'room_bed': room_bed,
            'room_bath': room_bath,
            'living_measure': living_measure,
            'lot_measure': lot_measure,
            'sight': 0,
            'quality': quality,
            'ceil_measure': 8 * room_bed,
            'basement': basement,
            'yr_renovated': yr_renovated,
            'zipcode': zipcode,
            'lat': 47.6,
            'living_measure15': living_measure,
            'lot_measure15': lot_measure,
            'furnished': furnished
        }

        input_df = pd.DataFrame([input_data])
        input_df[object_cols] = input_df[object_cols].astype(str)
        missing_cols = set(object_cols) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = ''

        encoded_input = OH_encoder.transform(input_df[object_cols])
        encoded_input_df = pd.DataFrame(encoded_input, columns=OH_encoder.get_feature_names_out(object_cols))

        final_input_df = pd.concat([input_df.drop(object_cols, axis=1), encoded_input_df], axis=1)

        for col in model.feature_names_:
            if col not in final_input_df.columns:
                final_input_df[col] = 0

        final_input_df = final_input_df[model.feature_names_]

        predicted_price = model.predict(final_input_df)[0]

        record = {
            "user_id": ObjectId(session['user_id']),
            "user_input": input_data,
            "predicted_price": predicted_price,
            "timestamp": datetime.datetime.now()
        }

        prediction_collection.insert_one(record)

        return render_template('predict.html', result=f"${predicted_price:,.2f}")

    return render_template('predict.html')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
