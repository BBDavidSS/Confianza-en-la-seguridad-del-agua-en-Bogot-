# 💧 Confianza en la Seguridad del Agua en Bogotá

> **Análisis Probabilístico y Estadístico basado en Machine Learning**  
> Informe inicial fundamentado en el artículo Scopus:  
> *"Water injustice in Colombia: Perceptions and realities of water quality examined through advanced machine learning"*

---

## 👤 Información del Estudiante

| Campo | Detalle |
|-------|---------|
| **Nombre** | Brayan David Santos Alvarez |
| **Código** | 20251020157 |
| **Programa** | Ingeniería Ambiental / Ciencias de la Salud |
| **Institución** | Universidad de Bogotá |

---

## 📋 Descripción del Proyecto

Este repositorio contiene el análisis estadístico y probabilístico de la **confianza ciudadana en la calidad del agua potable en Bogotá, D.C.**, Colombia. Se examinan variables fisicoquímicas reales reportadas por la EAAB (2022–2024), variables socioeconómicas (estrato) y datos de percepción ciudadana, aplicando técnicas de estadística inferencial y aprendizaje automático.

### 🎯 Pregunta de investigación

> ¿En qué medida la confianza ciudadana en la seguridad del agua en Bogotá se explica por variables objetivas de calidad fisicoquímica versus factores socioeconómicos y de percepción subjetiva?

---

## 📁 Estructura del Repositorio

```
📦 agua-bogota-confianza/
├── 📄 README.md                      ← Este archivo
├── 🐍 agua_bogota_analysis.py        ← Script principal de análisis en Python
├── 📑 agua_bogota_informe.pdf        ← Informe académico en LaTeX (normas APA)
├── 📊 agua_bogota_presentacion.pptx  ← Presentación de diapositivas (10 slides)
├── 🖼️ figura_agua_bogota.png         ← Panel de 6 gráficas estadísticas
└── 🖥️ pantallazo_python.png          ← Pantallazo del proceso digital en Python
```

---

## 🔬 Métodos Estadísticos Aplicados

| # | Método | Herramienta | Resultado |
|---|--------|-------------|-----------|
| 1 | Estadística descriptiva | `numpy`, `pandas` | Media, D.E., CV por parámetro |
| 2 | Prueba de normalidad (Shapiro-Wilk) | `scipy.stats` | p > 0.05 → distribución normal ✓ |
| 3 | Intervalos de confianza 95% | `scipy.stats.t.interval` | IC del IRCA por estrato |
| 4 | Correlación de Pearson | `scipy.stats.pearsonr` | r = 0.891 (estrato–confianza) |
| 5 | Regresión lineal simple | `scipy.stats.linregress` | R² = 0.794, p < 0.001 |
| 6 | ANOVA de un factor | `scipy.stats.f_oneway` | F = 0.354, p = 0.879 |
| 7 | Clasificador Naive Bayes | `sklearn.naive_bayes` | Exactitud = **96.0 %** |

---

## 📊 Variables Analizadas

### Fisicoquímicas (datos reales EAAB 2022–2024, Res. 2115/2007)

| Variable | Unidad | Media observada | Límite máximo | ¿Cumple? |
|----------|--------|-----------------|---------------|----------|
| Turbidez | NTU | 0.445 | 2.0 | ✅ |
| pH | — | 7.12 | 6.5 – 9.0 | ✅ |
| Cloro residual | mg/L | 0.705 | 2.0 | ✅ |
| Color aparente | UPC | 5.20 | 15.0 | ✅ |
| Conductividad | µS/cm | 142 | 1 000 | ✅ |
| **IRCA** | **%** | **33.82** | **< 5 (sin riesgo)** | ⚠️ Riesgo medio |

### Socioeconómicas y de percepción

- **Estrato socioeconómico** (1 al 6)
- **Nivel de confianza percibida** (escala 1–5)
- **Etiqueta binaria**: confía / no confía

---

## 🚀 Instalación y Uso

### Requisitos

```bash
Python >= 3.8
```

### Instalar dependencias

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

### Ejecutar el análisis

```bash
python agua_bogota_analysis.py
```

### Salida esperada

```
============================================================
ANÁLISIS ESTADÍSTICO - AGUA BOGOTÁ
============================================================

1. ESTADÍSTICA DESCRIPTIVA
2. PRUEBA DE NORMALIDAD (Shapiro-Wilk)
3. INTERVALOS DE CONFIANZA 95% - IRCA
4. CORRELACIÓN (Pearson r)
5. REGRESIÓN LINEAL (Estrato → Confianza)
6. ANOVA - IRCA por Estrato
7. CLASIFICADOR NAIVE BAYES (Machine Learning)

✅ Figura guardada: figura_agua_bogota.png
✅ Script completado exitosamente.
```

---

## 📈 Hallazgos Principales

```
🔵 Calidad objetiva:   El agua de Bogotá CUMPLE la normativa en todos los
                        parámetros fisicoquímicos monitoreados.

🟡 IRCA promedio:      33.82 % → Riesgo MEDIO técnico, pero HOMOGÉNEO
                        entre todos los estratos (ANOVA p = 0.879).

🔴 Brecha perceptual:  r(estrato, confianza) = 0.891 ★★★
                        El estrato explica el 79.4 % de la varianza
                        en confianza (R² = 0.794, p < 0.001).

🤖 Machine Learning:   Naive Bayes clasifica con 96 % de exactitud si un
                        ciudadano confía o no en el agua de Bogotá.
```

> **Conclusión central:** Existe una **inequidad hídrica de naturaleza perceptual** en Bogotá. La desconfianza en el agua no responde a diferencias técnicas reales, sino al estrato socioeconómico.

---

## 🗂️ Contenido Detallado

### `agua_bogota_analysis.py`
Script Python completo con los 7 procedimientos estadísticos. Genera automáticamente el panel de gráficas (`figura_agua_bogota.png`).

### `agua_bogota_informe.pdf`
Informe académico de 18 páginas compilado en LaTeX con:
- Normas APA 7ª edición
- Ecuaciones formales (distribución normal, IC, Pearson, regresión, Bayes)
- Tablas de resultados y referencias bibliográficas

### `agua_bogota_presentacion.pptx`
Presentación de 10 diapositivas que incluye:
- Metodología **manual** (paso a paso con fórmulas)
- Metodología **digital** (pantallazo del proceso en Python)
- Resultados, correlación, Naive Bayes y conclusiones

---

## 📚 Referencia Principal (Scopus)

> [Autor/es]. (2024). *Water injustice in Colombia: Perceptions and realities of water quality examined through advanced machine learning*. [Revista indexada en Scopus].

## 📖 Normativa

> Ministerio de Salud y Ministerio de Ambiente. (2007). *Resolución 2115 de 2007*. República de Colombia.

## 🏛️ Fuentes de datos

- **EAAB** — Empresa de Acueducto y Alcantarillado de Bogotá: https://www.acueducto.com.co
- **IDEAM** — Instituto de Hidrología, Meteorología y Estudios Ambientales
- **Secretaría Distrital de Salud de Bogotá**

---

## 📜 Licencia

Este proyecto es de uso académico. Los datos de la EAAB son de acceso público.

---

<div align="center">
  <sub>Elaborado por <strong>Brayan David Santos Alvarez</strong> · Código <strong>20251020157</strong></sub>
</div>
