# 🏡 Cotizador Inteligente de Viviendas (Real Estate Analytics)

Aplicación Web Full-Stack de estimación de precios inmobiliarios impulsada por **Machine Learning**. Diseñada para mostrar la integración óptima entre el análisis de datos, un backend robusto de alto rendimiento y una interfaz de usuario moderna y atractiva.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Vue.js](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vue.js&logoColor=4FC08D)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 🚀 Características Principales

- **Predicción en Tiempo Real**: Estima el valor de mercado de una propiedad en base a superficie, habitaciones, baños y zona.
- **Modelo de Machine Learning**: Regresión Lineal entrenada con Scikit-Learn con métricas de confianza transparentes.
- **Validación Estricta**: Control de tipos de datos en el backend mediante esquemas de **Pydantic**.
- **Interfaz Moderna (UI/UX)**: Desarrollada en **Vue 3 (Composition API)** con diseño futurista/minimalista usando **Tailwind CSS**, sliders interactivos y estados de carga fluidos.
- **Análisis Comparativo**: Muestra visualmente cómo se ubica el precio estimado de la propiedad frente al promedio de la zona.

---

## 📂 Estructura del Proyecto

Cotiza-viviendas/
├── BackEnd/
│   ├── main.py              # API FastAPI, entrenamiento y endpoint /predict
│   ├── requirements.txt     # Dependencias de Python
│   └── model.pkl            # Modelo entrenado de Scikit-Learn (generado automáticamente)
└── FrontEnd/
    ├── src/
    │   ├── assets/          # Estilos globales (Tailwind)
    │   ├── components/      # PropertyForm.vue y PropertyResults.vue
    │   ├── App.vue          # Componente principal y lógica de comunicación API
    │   └── main.js          # Punto de montaje de Vue
    └── package.json         # Dependencias de Node.js

🛠️Instalación y Ejecución Local

1. Clonar el repositorio
git clone https://github.com/tu-usuario/cotiza-viviendas.git
cd cotiza-viviendas

2. Configurar y ejecutar el Backend (FastAPI)
cd BackEnd
pip install -r requirements.txt
uvicorn main:app --reload

La API estará corriendo en http://localhost:8000 (puedes ver la documentación interactiva en /docs).
3. Configurar y ejecutar el Frontend (Vue 3)

Abre otra terminal:
cd FrontEnd
npm install
npm run dev

La aplicación web estará disponible en http://localhost:5173.

☁️ Despliegue en Producción

    Backend (Render): Sube la carpeta BackEnd, configura el comando de inicio (uvicorn main:app --host 0.0.0.0 --port $PORT) y añade las dependencias con requirements.txt.

    Frontend (Netlify): Sube la carpeta FrontEnd, configura el comando de compilación (npm run build) y la carpeta de publicación (dist). No olvides configurar la variable de entorno VITE_API_URL apuntando a tu URL pública de Render.
