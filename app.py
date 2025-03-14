import os
import json
import pytesseract
import numpy as np
import streamlit as st
import pandas as pd
import cv2
import re
import torch
from torch import nn
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from openfoodfacts import API  # Import the API class

# Streamlit page config
st.set_page_config(layout="wide", page_title="NutriVision: Allergy-Aware Recipe Generator", page_icon="🍳")

# Load environment variables and Groq client
load_dotenv()
api_key = os.getenv("GROQ_API")
client = Groq(api_key=api_key)

# Load allergy data
allergy_df = pd.read_csv("food_and_allergy.csv")

# Initialize Open Food Facts API with a user agent
off_api = API(user_agent="NutriVision by SahilDeogade sahil.deogade22@vit.edu")  # Replace with your info
# Set Tesseract path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Your existing generate_recipe function (unchanged except for profile integration)
def generate_recipe(query, dietary_pref, allergies, profile=None):
    if dietary_pref is None:
        dietary_pref = "None specified"
    
    forbidden_ingredients = []
    substitutes = []
    if allergies:
        allergy_list = [a.strip().lower() for a in allergies.split(",")]
        for allergy in allergy_list:
            matches = allergy_df[allergy_df["Allergy"].str.lower() == allergy]
            if not matches.empty:
                forbidden_ingredients.extend(matches["Food"].tolist())
                substitutes.extend([f"{food} → {sub}" for food, sub in zip(matches["Food"], matches["Substitute"])])

    allergy_note = ""
    if forbidden_ingredients:
        allergy_note = f"Avoid these ingredients due to allergies: {', '.join(forbidden_ingredients)}. Use these substitutes where applicable: {', '.join(substitutes)}."

    profile_note = "Based on your recent scans, I’ve tailored this recipe to balance your nutrition."
    profile_summary = ""
    if profile is not None:
        profile_summary = f"Profile suggests focus on: {'protein' if profile[0, 0] < 10 else 'balance'}."

    messages = [
        {"role": "system",
         "content": f'''
         You are NutriVision, a helpful AI chef. Your goal is to generate recipes tailored to the user's dietary preferences, allergies, and nutritional profile.
         - Exclude these ingredients due to allergies: {", ".join(forbidden_ingredients) if forbidden_ingredients else "None"}.
         - Use substitutes where applicable: {", ".join(substitutes) if substitutes else "None"}.
         - {profile_note} {profile_summary}
         - Format each recipe with:
           - Title: A creative name
           - Ingredients: List with precise metric measurements (e.g., 200g, 50ml)
           - Directions: Clear, numbered steps
           - Allergy Note: One sentence explaining how it accommodates the user's allergies
           - Nutritional Info: Per serving (calories, protein, fat, carbs)
           - Cooking Time: Estimated duration
         - Be polite and friendly, use a few emojis 😊🍳
         - If multiple dishes are requested (e.g., "pizza, tacos"), generate separate recipes for each, formatted as individual sections starting with "### [Dish] Product Recipe".
         '''},
        {"role": "user",
         "content": f"Generate a recipe for: {query}. Follow the dietary restrictions: {dietary_pref}. Handle allergies: {allergies if allergies else 'None specified'}. Provide measurements in the metric system."},
    ]
    progress_bar = st.progress(0)
    response_content = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )
    for i, chunk in enumerate(completion):
        response_content += chunk.choices[0].delta.content or ""
        progress = min((i + 1) / 20, 1.0)
        progress_bar.progress(progress)
    progress_bar.empty()
    return response_content

# Your existing parse_response function
def parse_response(response_text):
    recipes = []
    nutritional_info = []
    sections = response_text.split("### ")
    for section in sections:
        if not section.strip():
            continue
        if "Recipe" in section:
            recipes.append(f"### {section}")
        elif "Nutritional Info" in section:
            nutritional_info.append(section.strip())
    return recipes, nutritional_info

# Updated CV Function for Packet Scanning (Tesseract + Generic Open Food Facts Query)
def scan_packet(image_file):
    try:
        # Load and preprocess image
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)  # Load as BGR
        if img is None:
            raise ValueError("Failed to load image")
        
        # Convert to grayscale for OCR
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for OCR (adjust contrast, binarize)
        gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # OCR with Tesseract (using PSM 6 for sparse text like labels)
        text = pytesseract.image_to_string(gray_img, config='--psm 6')
        
        # Clean and normalize the text
        text = text.strip().replace("\n", " ").replace("\r", " ")
        
        # Generic extraction of product name (look for capitalized words or phrases after common keywords)
        potential_names = []
        lines = text.split(".")
        for line in lines:
            line = line.strip()
            if line:
                # Look for capitalized words or phrases after "Brand," "Product," or similar
                name_match = re.search(r"(?:Brand|Product|Name)\s*([A-Za-z\s]+)|([A-Za-z\s]+(?=[,\.!]))", line, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1) or name_match.group(2)
                    if name and len(name.split()) > 1:  # Prefer multi-word names
                        potential_names.append(name.strip())
        
        # Extract barcode (sequence of 8+ digits)
        barcode = re.search(r"\d{8,}", text)
        
        # Try Open Food Facts queries with potential names or barcode
        product_data = None
        if potential_names:
            for name in potential_names[:3]:  # Try up to 3 potential names
                cleaned_name = name.strip().replace("  ", " ")
                product_data = off_api.product(name=cleaned_name)
                if product_data and product_data.get("product"):
                    break
        elif barcode:
            product_data = off_api.product(barcode=barcode.group(0))
        
        if product_data and product_data.get("product"):
            nutrition = {
                "calories": product_data["product"].get("nutriments", {}).get("energy-kcal", 100),
                "fat": product_data["product"].get("nutriments", {}).get("fat", 0),
                "carbs": product_data["product"].get("nutriments", {}).get("carbohydrates", 0),
                "protein": product_data["product"].get("nutriments", {}).get("proteins", 0)
            }
            ingredients = product_data["product"].get("ingredients_text", "").split(",") if product_data["product"].get("ingredients_text") else ["unknown"]
        else:
            # Fallback to defaults if no Open Food Facts data
            nutrition = {"calories": 100, "fat": 0, "carbs": 0, "protein": 0}
            ingredients = ["unknown"]
            st.warning("Could not find product in Open Food Facts. Using default values.")
        
        return {"nutrition": nutrition, "ingredients": [i.strip().lower() for i in ingredients if i.strip()]}
    except Exception as e:
        st.error(f"Error scanning packet: {e}")
        return {"nutrition": {"calories": 100, "fat": 0, "carbs": 0, "protein": 0}, "ingredients": ["unknown"]}

def parse_nutrition(text):
    patterns = {
        "calories": r"(?:calories|energy):?\s*(\d+)",
        "fat": r"fat:?\s*(\d+\.?\d*)\s*(g|mg)",
        "carbs": r"carbo?hydrate?s?:?\s*(\d+\.?\d*)\s*(g|mg)",
        "protein": r"protein:?\s*(\d+\.?\d*)\s*(g|mg)"
    }
    nutrition = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            nutrition[key] = float(match.group(1))
    return nutrition

def parse_ingredients(text):
    ingr_section = re.search(r"ingredients:?\s*(.+)", text, re.IGNORECASE)
    if ingr_section:
        return [i.strip().lower() for i in ingr_section.group(1).split(",")]
    return []

# JSON Storage Functions
def load_scans():
    if os.path.exists("scans.json"):
        with open("scans.json", "r") as f:
            return json.load(f)
    return []

def save_scan(nutrition, ingredients):
    scans = load_scans()
    scan_data = {
        "timestamp": datetime.now().isoformat(),
        "nutrition": nutrition,
        "ingredients": ingredients
    }
    scans.append(scan_data)
    with open("scans.json", "w") as f:
        json.dump(scans, f, indent=2)

# LSTM Profiler
class NutritionProfiler(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 16)
    
    def forward(self, scan_history):
        out, _ = self.lstm(scan_history)
        return self.fc(out[:, -1, :])

profiler = NutritionProfiler()
if os.path.exists("profiler.pt"):
    profiler.load_state_dict(torch.load("profiler.pt"))

def update_profile(scans):
    if not scans:
        return None
    nutrition_data = [s["nutrition"] for s in scans]
    features = [[d.get("calories", 0), d.get("fat", 0), d.get("carbs", 0), d.get("protein", 0)] 
                for d in nutrition_data]
    scan_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return profiler(scan_tensor)

# Streamlit Tabs
tabs = st.tabs(["Recipe Generator", "Packet Profiler"])

# Tab 1: Recipe Generator
with tabs[0]:
    st.title("🍳 NutriVision: Allergy-Aware Recipe Generator")
    with st.sidebar:
        dietary_pref = st.text_input("Dietary Preferences", placeholder="e.g., vegetarian, vegan")
        allergies = st.text_input("Allergies", placeholder="e.g., nuts, dairy, gluten")
        if st.button("Clear Inputs"):
            st.session_state.clear()
            st.rerun()
    
    query = st.text_input("What would you like to cook?", placeholder="e.g., pizza, tacos, soup")
    if st.button("Generate Recipes"):
        if query.strip():
            with st.spinner("Generating your recipes..."):
                scans = load_scans()
                profile = update_profile(scans)
                recipe_text = generate_recipe(query, dietary_pref, allergies, profile)
                recipes, nutritional_info = parse_response(recipe_text)
                st.subheader("Your Recipes")
                for recipe in recipes:
                    with st.expander(f"📋 {recipe.splitlines()[0].replace('### ', '')}", expanded=False):
                        st.markdown(recipe)
                if nutritional_info:
                    st.subheader("Nutritional Info")
                    with st.expander("📊 Combined Nutritional Information", expanded=False):
                        st.markdown("---")
                        for info in nutritional_info:
                            st.markdown(info)
        else:
            st.error("Please enter what you’d like to cook!")

# Tab 2: Packet Profiler
with tabs[1]:
    st.title("📦 Packet Profiler")
    st.write("Upload a food packet image to analyze its nutrition and build your profile!")
    
    uploaded_file = st.file_uploader("Choose a packet image", type=["jpg", "png"])
    if uploaded_file:
        with st.spinner("Scanning your packet..."):
            scan_data = scan_packet(uploaded_file)
            save_scan(scan_data["nutrition"], scan_data["ingredients"])
            st.success("Scan complete!")
            st.subheader("Latest Scan")
            st.write("**Nutrition:**", scan_data["nutrition"])
            st.write("**Ingredients:**", ", ".join(scan_data["ingredients"]))
    
    # Display Profile Insights
    scans = load_scans()
    if scans:
        st.subheader("Your Nutritional Profile")
        profile = update_profile(scans)
        avg_nutrition = {
            "calories": sum(s["nutrition"].get("calories", 0) for s in scans) / len(scans),
            "fat": sum(s["nutrition"].get("fat", 0) for s in scans) / len(scans),
            "carbs": sum(s["nutrition"].get("carbs", 0) for s in scans) / len(scans),
            "protein": sum(s["nutrition"].get("protein", 0) for s in scans) / len(scans)
        }
        st.write("**Average Nutrition (per scan):**", avg_nutrition)
        frequent_ingredients = pd.Series([i for s in scans for i in s["ingredients"]]).value_counts().head(5)
        st.write("**Top 5 Frequent Ingredients:**", frequent_ingredients.to_dict())
        
        if profile is not None:
            st.write("**Profile Insight:**", "Focus on protein-rich foods" if avg_nutrition["protein"] < 10 else "Balanced diet detected")
        
        if st.button("Generate Recipe from Profile"):
            with st.spinner("Cooking up something special..."):
                recipe_text = generate_recipe("meal using recent scans", dietary_pref, allergies, profile)
                recipes, _ = parse_response(recipe_text)
                for recipe in recipes:
                    with st.expander(f"📋 {recipe.splitlines()[0].replace('### ', '')}", expanded=False):
                        st.markdown(recipe)

# CSS
st.markdown("""
    <style>
    .stButton button {background-color: #98C264; color: white; padding: 10px 20px; border-radius: 5px; border: none;}
    .stButton button:hover {background-color: #7A9C4F; color: #296232;}
    .stTextInput > div > div > input {border-radius: 5px;}
    .stTextInput > div > div > input:focus {border: 2px solid #285C34; box-shadow: 2px;}
    .stExpander {border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 20px;}
    .stExpander > label {font-weight: bold; font-size: 1.2em;}
    </style>
""", unsafe_allow_html=True)

# Optional: Keep CV dependencies commented for future use
# import cv2
# from ultralytics import YOLO