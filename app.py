from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo actuarial | K-Means", page_icon="📊", layout="centered")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def cargar_modelo():
    ruta = BASE / "kmeans_riesgo_actuarial.pkl"
    return joblib.load(ruta)

@st.cache_data
def cargar_metadata():
    with (BASE / "model_metadata.json").open(encoding="utf-8") as archivo:
        return json.load(archivo)

st.title("📊 Clasificación exploratoria de riesgo actuarial")
st.write("Modelo K-Means entrenado con información de seguros médicos. Introduce los datos de un cliente para identificar el grupo al que pertenece.")
st.info("Esta herramienta es una demostración académica. Los niveles Bajo, Medio y Alto se asignaron a los clusters según sus cargos médicos promedio; no representan una evaluación clínica ni una predicción validada de costos futuros.")

try:
    modelo = cargar_modelo()
    metadata = cargar_metadata()
except FileNotFoundError as exc:
    st.error(f"Falta un archivo necesario: {Path(exc.filename).name if exc.filename else exc}")
    st.stop()
except Exception as exc:
    st.error(f"No fue posible cargar el modelo o sus metadatos: {exc}")
    st.stop()

with st.form("evaluar"):
    age = st.number_input("Edad", min_value=0, max_value=120, value=30, step=1)
    sex = st.selectbox("Sexo (categoría del dataset)", ["female", "male"], format_func=lambda x: {"female":"Femenino", "male":"Masculino"}[x])
    bmi = st.number_input("Índice de masa corporal (BMI)", min_value=1.0, max_value=90.0, value=25.0, step=0.1)
    children = st.number_input("Número de hijos/dependientes", min_value=0, max_value=20, value=0, step=1)
    smoker = st.selectbox("¿Fuma?", ["no", "yes"], format_func=lambda x: "Sí" if x == "yes" else "No")
    region = st.selectbox("Región del dataset", ["northeast", "northwest", "southeast", "southwest"])
    charges = st.number_input("Cargos médicos observados (unidades del dataset)", min_value=0.0, value=5000.0, step=100.0)
    enviar = st.form_submit_button("Evaluar cliente", type="primary", use_container_width=True)

if enviar:
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region, "charges": charges
    }])
    try:
        cluster = int(modelo.predict(cliente)[0])
        mapa = metadata.get("mapa_riesgo", {})
        riesgo = mapa.get(str(cluster), "Sin etiqueta")
        explicaciones = {
            "Bajo": "Perfil agrupado con clientes de menor costo médico promedio en el conjunto de entrenamiento.",
            "Medio": "Perfil agrupado con clientes de costo médico promedio intermedio en el conjunto de entrenamiento.",
            "Alto": "Perfil agrupado con clientes de mayor costo médico promedio en el conjunto de entrenamiento."
        }
        st.success(f"Grupo asignado: {cluster} · Riesgo actuarial exploratorio: {riesgo}")
        st.write(explicaciones.get(riesgo, "Revisa el mapa de riesgos del archivo JSON."))
        st.dataframe(cliente, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"No se pudo generar la clasificación: {exc}")

with st.expander("Información del entrenamiento"):
    st.write("**Modelo:**", metadata.get("nombre_modelo", "K-Means"))
    st.write("**Número de clusters:**", metadata.get("n_clusters", "No disponible"))
    st.write("**Silhouette score:**", metadata.get("silhouette_score", "No disponible"))
    st.write("**Kernels SVM entrenados en el notebook:**", ", ".join(metadata.get("svm_kernels", [])))
    st.caption("El notebook entrena SVM sobre componentes PCA; esta interfaz usa K-Means directamente para clientes nuevos porque el PCA entrenado no se guardó en el notebook original.")

