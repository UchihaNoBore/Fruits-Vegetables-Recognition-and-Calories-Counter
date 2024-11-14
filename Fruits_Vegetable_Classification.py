import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
from nutrition_db import nutrition_db

model = load_model('final.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def get_nutrition_info(prediction):
    """Get nutrition information from local database"""
    prediction = prediction.lower()
    if prediction in nutrition_db:
        return nutrition_db[prediction]
    return None

def display_nutrition_info(nutrition_data):
    """Display nutrition information in a formatted way"""
    if nutrition_data:
        st.subheader("Nutrition Facts (per 100g)")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Column 1: Basic nutrition facts
        with col1:
            st.markdown("**Basic Nutrition:**")
            st.write(f"🔸 Calories: {nutrition_data['calories']}")
            st.write(f"🔸 Carbohydrates: {nutrition_data['carbs']}")
            st.write(f"🔸 Fiber: {nutrition_data['fiber']}")
            st.write(f"🔸 Sugar: {nutrition_data['sugar']}")
            st.write(f"🔸 Protein: {nutrition_data['protein']}")
        
        # Column 2: Vitamins and minerals
        with col2:
            st.markdown("**Vitamins:**")
            for vitamin in nutrition_data['vitamins']:
                st.write(f"🔸 {vitamin}")
            
            st.markdown("**Minerals:**")
            for mineral in nutrition_data['minerals']:
                st.write(f"🔸 {mineral}")

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

def run():
    st.title("Fruits🍍-Vegetable🍅 Recognition & Nutrition Guide")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            
            # Display category
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            
            # Display prediction
            st.success("**Predicted : " + result + '**')
            
            # Get and display nutrition information
            nutrition_data = get_nutrition_info(result)
            if nutrition_data:
                display_nutrition_info(nutrition_data)
            else:
                st.warning(f"Nutrition information for {result} is not available in our database yet.")

if __name__ == "__main__":
    run()