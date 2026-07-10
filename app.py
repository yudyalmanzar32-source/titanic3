import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image, ImageDraw

# Set up page configurations
st.set_page_config(
    page_title="Titanic Predictor & CV Hub",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Glassmorphism + Dark Mode)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f1f5f9;
}

/* Sidebar theme */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Custom premium card */
.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 15px 40px 0 rgba(6, 182, 212, 0.15);
}

.metric-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #94a3b8;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.gradient-text-blue {
    background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.gradient-text-green {
    background: linear-gradient(90deg, #4ade80 0%, #16a34a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.gradient-text-red {
    background: linear-gradient(90deg, #f87171 0%, #dc2626 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 10px;
}

.badge-survived {
    background-color: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.badge-died {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Header style */
.header-container {
    padding: 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 30px;
}

.main-title {
    font-size: 40px;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 16px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ----------------- AUXILIARY FUNCTIONS -----------------

def load_models():
    """Load trained models and metadata from local files, or train them on-the-fly as fallback."""
    try:
        with open('model_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('model_regressor.pkl', 'rb') as f:
            reg = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
        return clf, reg, meta
    except Exception as e:
        # Fallback: Train models dynamically if pickles fail (e.g., version mismatch on cloud environments)
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            csv_path = 'titanic.csv'
            if not os.path.exists(csv_path):
                # Try locating relative to the script location
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, 'titanic.csv')
                
            if not os.path.exists(csv_path):
                st.error(f"Error: titanic.csv no encontrado en '{os.path.abspath('titanic.csv')}' ni en '{csv_path}'.")
                return None, None, None
                
            df = pd.read_csv(csv_path)
            
            # Impute missing values
            age_median = df['Age'].median()
            df['Age'] = df['Age'].fillna(age_median)
            
            embarked_mode = df['Embarked'].mode()[0]
            df['Embarked'] = df['Embarked'].fillna(embarked_mode)
            
            fare_median = df['Fare'].median()
            df['Fare'] = df['Fare'].fillna(fare_median)
            
            # Encode categoricals
            df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
            df['Embarked_encoded'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
            
            # Train Logistic Regression
            X_clf = df[['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']]
            y_clf = df['Survived']
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_clf, y_clf)
            
            # Train Linear Regression
            X_reg = df[['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Embarked_encoded', 'Survived']]
            y_reg = df['Fare']
            reg = LinearRegression()
            reg.fit(X_reg, y_reg)
            
            meta = {
                'age_median': age_median,
                'fare_median': fare_median,
                'embarked_mode': embarked_mode
            }
            return clf, reg, meta
        except Exception as inner_e:
            st.error(f"Error durante el entrenamiento de fallback: {inner_e}")
            st.exception(inner_e)
            return None, None, None

def simulate_cv_detection(image):
    """Draw bounding boxes and count passengers on an uploaded image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    
    # Predefined bounding boxes relative to image sizes for simulation
    # This represents a realistic passenger face detection scenario
    boxes = [
        (0.18, 0.22, 0.32, 0.44),
        (0.38, 0.18, 0.52, 0.41),
        (0.58, 0.28, 0.70, 0.52),
        (0.74, 0.20, 0.88, 0.45)
    ]
    
    for idx, (x1_f, y1_f, x2_f, y2_f) in enumerate(boxes):
        x1, y1 = int(x1_f * width), int(y1_f * height)
        x2, y2 = int(x2_f * width), int(y2_f * height)
        
        # Cyan color outline
        draw.rectangle([x1, y1, x2, y2], outline="#06b6d4", width=max(3, int(width/200)))
        
        # Label container
        label_height = max(18, int(height/25))
        draw.rectangle([x1, y1 - label_height, x1 + int(width/8), y1], fill="#06b6d4")
        
        # Note: default font is basic but guaranteed to work without file issues
        draw.text((x1 + 4, y1 - label_height + 2), f"ID #{idx+1}", fill="black")
        
    return img_draw, len(boxes)

# ----------------- APP HEADER -----------------

st.markdown("""
<div class="header-container">
    <div class="main-title">🚢 Inteligencia a Bordo: Dashboard Titanic & Visión por Computadora</div>
    <div class="subtitle">Metodología CRISP-ML(Q) aplicada para la predicción de supervivencia, simulación de costos y monitoreo visual.</div>
</div>
""", unsafe_allow_html=True)

# Load the models
clf, reg, meta = load_models()

if clf is None or reg is None or meta is None:
    st.error("⚠️ No se encontraron los modelos entrenados. Por favor, asegúrate de que el script de entrenamiento se ejecute con éxito para generar `model_classifier.pkl` y `model_regressor.pkl`.")
    st.info("💡 Puedes entrenar los modelos ejecutando el cuaderno de Jupyter o ejecutando el script `train_models.py` en tu terminal.")
    st.stop()

# ----------------- SIDEBAR CONFIGURATION -----------------

st.sidebar.header("⚙️ Configuración del Pasajero")
st.sidebar.markdown("Define los atributos para calcular la probabilidad de supervivencia y predecir el costo del pasaje.")

pclass_label = st.sidebar.selectbox(
    "Clase del Boleto (Pclass)",
    ["Primera Clase", "Segunda Clase", "Tercera Clase"]
)
pclass_map = {"Primera Clase": 1, "Segunda Clase": 2, "Tercera Clase": 3}
pclass = pclass_map[pclass_label]

sex_label = st.sidebar.selectbox("Género", ["Femenino", "Masculino"])
sex_encoded = 1 if sex_label == "Femenino" else 0

age = st.sidebar.number_input(
    "Edad (Años)",
    min_value=1,
    max_value=100,
    value=int(meta['age_median'])
)

sibsp = st.sidebar.slider("Hermanos / Cónyuges a bordo (SibSp)", 0, 8, 0)
parch = st.sidebar.slider("Padres / Hijos a bordo (Parch)", 0, 6, 0)

embarked_label = st.sidebar.selectbox(
    "Puerto de Embarque",
    ["Cherburgo (C)", "Queenstown (Q)", "Southampton (S)"]
)
embarked_map = {"Cherburgo (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
embarked_encoded = embarked_map[embarked_label]

# ----------------- MAIN LAYOUT -----------------

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("🔮 Modelado Predictivo e Inferencia")
    
    # 2-Step Prediction Loop to handle interdependency
    # 1. Estimate initial survival state using median fare
    initial_fare = meta['fare_median']
    initial_features = [[pclass, sex_encoded, age, sibsp, parch, initial_fare, embarked_encoded]]
    initial_survived = int(clf.predict(initial_features)[0])
    
    # 2. Predict Fare using the estimated survival state
    reg_features = [[pclass, sex_encoded, age, sibsp, parch, embarked_encoded, initial_survived]]
    predicted_fare = max(0.0, float(reg.predict(reg_features)[0]))
    
    # 3. Re-run classifier with the predicted Fare to get final probability
    final_features = [[pclass, sex_encoded, age, sibsp, parch, predicted_fare, embarked_encoded]]
    survival_prob = clf.predict_proba(final_features)[0][1]
    survival_pred = clf.predict(final_features)[0]
    
    # Layout cards
    prob_percentage = int(survival_prob * 100)
    
    if survival_pred == 1:
        text_color_class = "gradient-text-green"
        badge_html = f'<div class="status-badge badge-survived">Predicción: SOBREVIVE ({prob_percentage}%)</div>'
    else:
        text_color_class = "gradient-text-red"
        badge_html = f'<div class="status-badge badge-died">Predicción: NO SOBREVIVE ({prob_percentage}%)</div>'
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Clasificador: Probabilidad de Supervivencia</div>
        <div class="metric-value {text_color_class}">Probabilidad: {prob_percentage}%</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Regresor: Costo del Pasaje Predicho (Fare)</div>
        <div class="metric-value gradient-text-blue">${predicted_fare:.2f} USD</div>
        <div style="font-size:13px; color:#94a3b8; margin-top:8px;">
            Predicción basada en un modelo de Regresión Lineal multivariable.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Detalle de la Metodología de Inferencia (Bucle de Retroalimentación)"):
        st.write("""
        Las variables **Survived** y **Fare** son interdependientes en nuestro modelado:
        1. **Fase A (Clasificación Inicial):** Calculamos la supervivencia estimada utilizando la tarifa mediana de la población del Titanic.
        2. **Fase B (Regresión Lineal):** Predecimos la tarifa personalizada (`Fare`) usando la supervivencia estimada en la Fase A y los demás datos demográficos.
        3. **Fase C (Clasificación Final):** Recalculamos la probabilidad de supervivencia definitiva ingresando la tarifa calculada en la Fase B.
        Este bucle dinámico garantiza la consistencia física y económica entre ambos modelos.
        """)

with col2:
    st.subheader("👁️ Visión por Computadora (Simulada)")
    st.write("Sube una imagen o usa la demostración precargada para simular el conteo de pasajeros en una cámara de embarque con inteligencia artificial.")
    
    # Image selection
    cv_option = st.radio("Origen de la Imagen", ["Usar Imagen Demostrativa de RA", "Subir una Foto Propia"])
    
    uploaded_image = None
    
    if cv_option == "Usar Imagen Demostrativa de RA":
        sample_path = os.path.join("assets", "passenger_detector.png")
        # Fallback to general passenger_detector_ar if not renamed yet
        if not os.path.exists(sample_path):
            sample_path = os.path.join("assets", "passenger_detector_ar.png")
            
        if os.path.exists(sample_path):
            uploaded_image = Image.open(sample_path)
        else:
            st.warning("⚠️ No se encontró la imagen demostrativa en `assets/passenger_detector_ar.png`. Por favor, sube una foto propia.")
    else:
        uploaded_file = st.file_uploader("Elige una foto...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            
    if uploaded_image is not None:
        # Convert to RGB if palette image
        if uploaded_image.mode not in ('RGB', 'RGBA'):
            uploaded_image = uploaded_image.convert('RGB')
            
        with st.spinner("Procesando imagen con modelo de detección..."):
            processed_img, count = simulate_cv_detection(uploaded_image)
            
        st.image(processed_img, caption="Cámara de IA: Detección y Conteo Facial Activo", use_column_width=True)
        st.success(f"🤖 Detección finalizada: Se detectaron **{count}** pasajeros listos para abordar.")
    else:
        st.info("Carga una imagen en formato JPG o PNG para visualizar la detección digital.")
