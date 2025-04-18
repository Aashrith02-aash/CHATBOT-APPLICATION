from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import pandas as pd
from difflib import get_close_matches
import re
import requests
import uuid
import os
import string
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.secret_key = "your_secret_key"
bcrypt = Bcrypt(app)

# 🔹 MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://AASHRITH:ZTqfAYX7HV5zbBXE@cluster0.xhvxi.mongodb.net/chatbot_db?retryWrites=true&w=majority"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chatbot_db"]
users_collection = db["users"]
chats_collection = db["chats"]  # Collection for storing chat history

# 🔹 Load CSV Data
csv_file = r"C:\Users\vudat\CHATBOT\dataset.csv"  # Use raw string (r"") to avoid escape issues

def load_csv():
    """Loads data from CSV file into a dictionary with proper handling of missing values"""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, usecols=[0, 1], names=["question", "answer"], header=None, encoding="utf-8")
        df = df.dropna()  # Remove rows with missing questions or answers
        df["question"] = df["question"].astype(str).str.strip().str.lower()
        df["answer"] = df["answer"].astype(str).str.strip()
        return dict(zip(df["question"], df["answer"]))  # Create dictionary from CSV
    return {}

qa_dict = load_csv()

# 🔹 GPT-2 Model Initialization
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 Normalize input by removing punctuation and converting to lowercase
def normalize_text(text):
    text = text.lower().strip()
    return text.translate(str.maketrans("", "", string.punctuation))

# 🔹 Math Calculation (Basic using eval)
def calculate(expression):
    try:
        expression = re.sub(r'[^0-9+\-*/(). ]', '', expression)
        return eval(expression)
    except:
        return "Invalid math expression!"

# 🔹 GPT-2 Response Generation
def generate_gpt2_response(prompt):
    """Generates a response using GPT-2"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔹 Landing Page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/landing')
def landing():
    return render_template("landing.html")

# 🔹 Signup Route
@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")

        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return render_template("signup.html", error="User already exists!")

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        users_collection.insert_one({"email": email, "password_hash": hashed_password})
        return redirect(url_for("login"))

    return render_template("signup.html")

# 🔹 Login Route
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")

        user = users_collection.find_one({"email": email})
        if user and bcrypt.check_password_hash(user["password_hash"], password):
            session["user"] = email
            return redirect(url_for("chatbot"))
        else:
            return render_template("login.html", error="Invalid login. Please try again.")

    return render_template("login.html")

# 🔹 Chatbot Page (Shows Old Chats + New Chat Option)
@app.route('/chatbot')
def chatbot():
    if "user" not in session:
        return redirect(url_for("login"))

    user_email = session["user"]
    chats = list(chats_collection.find({"user_email": user_email}, {"_id": 0, "chat_id": 1}))

    return render_template("chatbot.html", chats=chats)

# 🔹 Create a New Chat Session
@app.route('/new_chat', methods=['POST'])
def new_chat():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_email = session["user"]
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID

    chat_data = {"chat_id": chat_id, "user_email": user_email, "messages": []}
    chats_collection.insert_one(chat_data)  # Save to MongoDB

    return jsonify({"message": "New chat started", "chat_id": chat_id})

# 🔹 Store and Retrieve Messages
@app.route('/get_response', methods=['POST'])
def get_response():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    global qa_dict  # Reload data dynamically
    qa_dict = load_csv()

    data = request.json
    chat_id = data.get("chat_id")
    user_message = data.get("message", "").strip()

    if not chat_id or not user_message:
        return jsonify({"error": "Chat ID and message required"}), 400

    chat = chats_collection.find_one({"chat_id": chat_id})
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    # Normalize user input
    normalized_message = normalize_text(user_message)

    # Store user message
    new_message = {
        "sender": "user",
        "message": user_message,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_collection.update_one({"chat_id": chat_id}, {"$push": {"messages": new_message}})

    # Generate bot response
    bot_reply_text = qa_dict.get(normalized_message, None)

    if not bot_reply_text:
        closest_match = get_close_matches(normalized_message, qa_dict.keys(), n=1, cutoff=0.7)
        if closest_match:
            bot_reply_text = qa_dict[closest_match[0]]
        else:
            bot_reply_text = generate_gpt2_response(user_message)

    # Store bot reply
    bot_reply = {
        "sender": "bot",
        "message": bot_reply_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_collection.update_one({"chat_id": chat_id}, {"$push": {"messages": bot_reply}})

    return jsonify({"response": bot_reply_text})

# 🔹 Logout Route
@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
