import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st
import tempfile
from sklearn.cluster import KMeans
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('C:/Users/rgbha/OneDrive/Desktop/season_model.h5')

# Define categories and palettes
categories = ["Deep Winter","Bright Winter","Cool Winter", "Light Spring","Warm Spring","Bright Spring","Light Summer","Soft Summer", "Cool Summer","Deep Autumn","Warm Autumn","Soft Autumn"]
season_palettes = {
    "Deep Winter": {
        "recommended": [
            ("Burgundy", "#800020"), ("Deep Teal", "#003333"), ("Charcoal", "#36454F"),
            ("Dark Chocolate", "#3B2F2F"), ("Pine Green", "#01796F"), ("Black", "#000000"),
            ("Wine Red", "#722F37"), ("Plum", "#673147")
        ],
        "to_avoid": [
            ("Pastels", "#FFCCCB"), ("Light Beige", "#F5F5DC"), ("Orange", "#FFA500"),
            ("Warm Yellow", "#FFD700"), ("Olive", "#808000")
        ]
    },
    "Bright Winter": {
        "recommended": [
            ("Fuchsia", "#FF00FF"), ("Icy Blue", "#AFDBF5"), ("Royal Blue", "#4169E1"),
            ("Bright White", "#FFFFFF"), ("Cherry Red", "#DE3163"), ("Cobalt", "#0047AB"),
            ("Emerald Green", "#50C878"), ("Vivid Purple", "#8F00FF")
        ],
        "to_avoid": [
            ("Earthy Tones", "#BC8F8F"), ("Taupe", "#483C32"), ("Warm Brown", "#8B4513"),
            ("Soft Yellow", "#FFFACD")
        ]
    },
    "Cool Winter": {
        "recommended": [
            ("Pure White", "#FFFFFF"), ("Navy", "#000080"), ("Cool Grey", "#8C92AC"),
            ("Icy Pink", "#FFD1DC"), ("Sapphire Blue", "#0F52BA"), ("Magenta", "#FF00FF"),
            ("True Red", "#FF0000"), ("Steel Blue", "#4682B4")
        ],
        "to_avoid": [
            ("Warm Tones", "#FFCC99"), ("Beige", "#F5F5DC"), ("Gold", "#FFD700"),
            ("Orange", "#FFA500"), ("Rust", "#B7410E")
        ]
    },
    "Light Spring": {
        "recommended": [
            ("Light Peach", "#FFDAB9"), ("Soft Yellow", "#FFFFE0"), ("Baby Pink", "#FFE4E1"),
            ("Aqua", "#00FFFF"), ("Mint Green", "#98FB98"), ("Light Coral", "#F08080"),
            ("Lavender", "#E6E6FA"), ("Soft Gold", "#DAA520")
        ],
        "not_recommended": [
            ("Black", "#000000"), ("Navy", "#000080"), ("Deep Grey", "#696969"), ("Icy Blue", "#87CEFA")
        ]
    },
    "Warm Spring": {
        "recommended": [
            ("Apricot", "#FBCEB1"), ("Warm Pink", "#FFC0CB"), ("Golden Yellow", "#FFD700"),
            ("Lime Green", "#32CD32"), ("Coral", "#FF7F50"), ("Turquoise", '#40E0D0'),
            ("Camel", '#C19A69'), ("Bright Orange", '#FFA500')
        ],
        "not_recommended": [
            ("Icy Pastels", "#E6E6FA"), ("Blue Undertones", "#0000FF")
        ]
    },
    "Bright Spring": {
        "recommended": [
            ("Hot Pink", "#FF69B4"), ("Bright Lime Green", '#00FF00'), ("Vivid Turquoise", '#00CED1'),
            ("Sunny Yellow", '#FFFF00'), ("Bright Coral", '#FF7F50'), ("Warm Red", '#DC143C'),
            ("Electric Blue", '#007FFF'), ("Bright Orange", '#FFA500')
        ],
        "not_recommended": [
            ("Muted Tones", "#D3D3D3"), ("Soft Pastels", '#E6E6FA'), ("Overly Dark Colors", "#000000")
        ]
    },
    "Light Summer": {
        "recommended": [
            ("Soft Lavender", '#E6E6FA'), ("Baby Blue", '#ADD8E6'), ("Powder Pink", '#FFB6C1'),
            ("Soft Mint", '#98FB98'), ("Dusty Rose", '#E1B49E'), ("Pale Grey", '#D3D3D3'),
            ("Light Periwinkle", '#CCCCFF'), ("Blush", '#FFDAB9')
        ],
        "not_recommended": [
            ("Orange", '#FFA500'), ("Mustard", '#DAA520'), ("Dark Colors", "#000000")
        ]
    },
    "Soft Summer": {
        "recommended": [
            ("Dusty Rose", '#E1B49E'), ("Taupe", '#D2B48C'), ("Muted Blue", '#87CEEB'),
            ("Sage Green", "#98FB98"), ("Soft Lavender", '#E6E6FA'), ("Cool Grey", '#808080'),
            ("Mauve", '#E0B0FF'), ("Muted Teal", '#4682B4')
        ],
        "not_recommended": [
             ("Pure White", '#FFFFFF'),
            ("Black", '#000000'), ("Vivid Red", '#FF0000')
        ]
    },
    "Cool Summer": {
        "recommended": [
            ("Soft Teal", '#4682B4'), ("Ice Pink", '#FFD1DC'), ("Soft Grey", '#D3D3D3'),
            ("Periwinkle", '#CCCCFF'), ("Rose", '#FFC0CB'), ("Blueberry", '#8A2BE2'),
            ("Dusty Blue", '#87CEEB'), ("Pale Silver", '#C0C0C0')
        ],
        "not_recommended": [
            ("Rust", '#B7410E'), ("Bright orange", '#FF7F00'), ("Golden yellow", '#FFD700'),
        
        ]
    },
    "Deep Autumn": {
        "recommended": [
            ("Dark olive", '#3C4134'), ("Auburn", '#A52A2A'), ("Deep burgundy", '#800020'),
            ("Chocolate brown", '#5C3317'), ("Dark mustard", '#CDA343'), ("Forest green", '#228B22'),
            ("Deep teal", '#008080'), ("Rust", '#B7410E')
        ],
        "not_recommended": [
            ("Icy blue", '#87CEFA'), ("Pinks", '#FFC0CB'), ("Stark white", '#FFFFFF')
        ]
    },
    "Warm Autumn": {
        "recommended": [
            ("Pumpkin", "#FBCEB1"), ("Warm tan", "#D2B48C"), ("Olive green", "#808000"),
            ("Golden brown", "#A0522D"), ("Burnt orange", "#CC5500"), ("Terracotta", "#E2725B"),
            ("Mustard yellow", "#DAA520"), ("Camel", "#C19A6B")
        ],
        "not_recommended": [
            ("Icy pink", "#FFD1DC"), ("Blue", "#0000FF"), ("Grey", "#808080")
        ]
    },
    "Soft Autumn": {
        "recommended": [
            ("Soft khaki", "#BDB76B"), ("Sage green", "#9CBA7F"), ("Muted gold", "#DAA520"),
            ("Dusty peach", "#FFDAB9"), ("Soft olive", "#556B2F"), ("Warm taupe", "#D2B48C"),
            ("Camel", "#C19A6B"), ("Warm grey", "#BEBEBE")
        ],
        "not_recommended": [
            ("Bright red", "#FF0000"), ("Cobalt blue", "#0047AB"), ("Icy pink", "#FFD1DC")
        ]
    }
}


def extract_dominant_colors(image, n_colors=3):
    # Resize and reshape image to extract colors (without displaying them)
    image = cv2.resize(image, (224, 224))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

# Streamlit app setup
# Streamlit app setup
st.title("Personal Color Analysis App")
st.write("Upload your image to find your color season!")
st.write("It is advised to upload an image with good lighting and a white background")
st.write("Please upload your Image without wearing specs or accesories.")
img = Image.open("C:/Users/rgbha/OneDrive/Desktop/Seasons/bright winter/bright_winter_38.png")
img = img.resize((175, 225))  # Define width and height in pixels

# Display the resized image
st.image(img, caption="Example of how the uploaded image should look")

# Image upload section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Process the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (224, 224))
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Detect colors without displaying them
    dominant_colors = extract_dominant_colors(image)

    # Prediction section
    image_array = np.expand_dims(image_resized, axis=0)  # Prepare image for model
    prediction = model.predict(image_array)
    predicted_class = categories[np.argmax(prediction)]
    
    # Submit button and results page
    if st.button("Submit"):
        st.session_state['predicted_class'] = predicted_class  # Store result in session state
        st.experimental_set_query_params(rerun=True) 

# Results page after clicking submit
if 'predicted_class' in st.session_state:
    predicted_class = st.session_state['predicted_class']
    st.write(f"You belong to this Season: **{predicted_class}**")

    # Function to create a color image
    def create_color_image(hex_code, size=(50, 50)):
        color_image = Image.new("RGB", size, hex_code)
        return color_image

    # Display recommended and not recommended colors in horizontal rows
    palette_data = season_palettes.get(predicted_class, {})
    if palette_data:
        st.write("Recommended Colors:")
        
        # Display recommended colors in a single row
        recommended_colors = palette_data.get("recommended", [])
        cols = st.columns(len(recommended_colors))
        for idx, (color_name, color_hex) in enumerate(recommended_colors):
            color_image = create_color_image(color_hex)
            with cols[idx]:
                st.image(color_image, caption=color_name, width=75)

        st.write("Not Recommended Colors:")

        # Display not recommended colors in a single row
        not_recommended_colors = palette_data.get("to_avoid", palette_data.get("not_recommended", []))
        cols = st.columns(len(not_recommended_colors))
        for idx, (color_name, color_hex) in enumerate(not_recommended_colors):
            color_image = create_color_image(color_hex)
            with cols[idx]:
                st.image(color_image, caption=color_name, width=75)
