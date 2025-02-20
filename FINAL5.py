import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template 
from typing import List, Dict
import google.generativeai as genai
import faiss  
import os 
from sentence_transformers import SentenceTransformer 
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-pro")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

def load_data(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Prerequisites"] = df["Prerequisites"].fillna("")
    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[["course_title", "course_organization", "course_Certificate_type", "course_rating", "course_difficulty", "course_students_enrolled", "Prerequisites"]]

def create_embeddings(text_list: List[str]) -> List[List[float]]:
    try:
        return embedding_model.encode(text_list).tolist()
    except Exception:
        return None

def store_embeddings(embeddings: List[List[float]], vector_db):
    vector_db.add(np.array(embeddings).astype("float32"))

def load_vector_db(path: str):
    try:
        index = faiss.read_index(path)
        return index
    except RuntimeError:
        return None

def create_user_profile(user_data: Dict) -> str:
    return f"Interests: {user_data['interests']}. Skills: {user_data['skills']}. Goals: {user_data['goals']}. Education: {user_data['education']}."

def vector_search(query_embedding: List[float], vector_db, top_k: int = 5) -> List[int]:
    query_embedding = np.array([query_embedding]).astype("float32").reshape(1, -1)
    _, I = vector_db.search(query_embedding, top_k)
    return I[0].tolist()

def get_prerequisites(course_id: int, course_data: pd.DataFrame) -> List[str]:
    return [p.strip() for p in course_data.loc[course_id, "Prerequisites"].split(",")]

def create_prompt(user_profile: str, relevant_courses: pd.DataFrame, augmented_data: Dict) -> str:
    return f"""
    Recommend courses to a user with the following profile: {user_profile}.
    The following courses are potentially relevant: {relevant_courses.to_string()}.
    Additional data about these courses: {augmented_data}.
    Provide a concise and personalized recommendation.
    """

def generate_response(prompt: str) -> str:
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return "Sorry, I could not generate a recommendation at this time."

def format_recommendation(response: str) -> str:
    return f"Here's a course recommendation for you:\n{response}"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_course():
    user_data = request.json
    user_profile = create_user_profile(user_data)
    profile_embedding = create_embeddings([user_profile])[0]
    relevant_course_ids = vector_search(profile_embedding, index)
    relevant_courses = df.loc[relevant_course_ids]
    augmented_data = {course_id: {"prerequisites": get_prerequisites(course_id, df)} for course_id in relevant_course_ids}
    prompt = create_prompt(user_profile, relevant_courses, augmented_data)
    recommendation = generate_response(prompt)
    return jsonify({"recommendation": format_recommendation(recommendation)})

@app.route("/vector-db", methods=["POST"])
def load_vector_db_api():
    try:
        global index
        index = load_vector_db(request.form["vector_db_path"])
        return jsonify({"message": "VectorDB loaded successfully!", "num_vectors": index.ntotal}) if index else jsonify({"message": "VectorDB loading failed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    csv_file_path = "nlp_test_data_final.csv"
    df = load_data(csv_file_path)
    if df is None:
        exit()
    df = clean_data(df)
    df = prepare_data(df)
    course_descriptions = df["course_title"].tolist()
    vector_db_path = "course_embeddings.faiss"
    if not os.path.exists(vector_db_path):
        embeddings = create_embeddings(course_descriptions)
        if embeddings is None:
            exit()
        d = len(embeddings[0])
        index = faiss.IndexFlatL2(d)
        store_embeddings(embeddings, index)
        faiss.write_index(index, vector_db_path)
    index = load_vector_db(vector_db_path)
    if index is None:
        exit()
    app.run(debug=True, use_reloader=False)
