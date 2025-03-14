# NutriVision

**NutriVision** is a Streamlit-based application designed to help users generate personalized recipes and track their nutritional intake. It offers two core features:

- **Recipe Generator**: Creates tailored recipes based on dietary preferences and allergies using Groq's Llama-3 model.
- **Packet Profiler**: Scans food packet images to extract nutritional information using Tesseract OCR and the Open Food Facts API.

Additionally, it includes a nutritional profiling feature that analyzes scan history to suggest balanced meals, enhancing personalization.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed (ensure the path is configured in `app.py`)
- A Groq API key for recipe generation

---

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sahildeogade03/NutriVision.git
   cd NutriVision

2. Create a .env file in the root directory with your Groq API key:
    ```bash
    GROQ_API=your_groq_api_key

3. **Running the Application**
-  Start the app with:
   ```bash
   streamlit run app.py

---

![Screenshot (34)](https://github.com/user-attachments/assets/e398c8a2-d4af-46d7-805c-0465e0506e0c)

![Screenshot (35)](https://github.com/user-attachments/assets/a03b3167-4de5-4182-9531-17385fff5174)

## Usage

**Recipe Generator**
-  Go to the "Recipe Generator" tab.
-  Enter a dish name (e.g., "pizza") in the input field.
-  Set dietary preferences (e.g., "vegan") and allergies (e.g., "gluten, peanuts") in the sidebar.
-  Click "Generate Recipes" to see customized recipes.
-  Use "Clear Inputs" to reset the fields.

**Packet Profiler**
-  Switch to the "Packet Profiler" tab.
-  Upload a food packet image (e.g., JPG, PNG).
-  View extracted nutritional details (calories, fat, carbs, ingredients).
-  Check your nutritional profile, including averages and frequent ingredients.
-  Click "Generate Recipe from Profile" for a meal based on your scans.

---
