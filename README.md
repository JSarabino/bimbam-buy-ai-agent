# BimBam Assistant

Agente de inteligencia artificial basado en **RAG** (*Retrieval-Augmented Generation*) para consultar las políticas y documentos corporativos de BimBam Buy mediante preguntas en lenguaje natural.

## Descripción

BimBam Buy es un e-commerce multiplataforma enfocado en ofrecer una experiencia de compra digital ágil, segura y orientada al cliente.

El objetivo del proyecto es construir un asistente que pueda consultar los documentos internos de la empresa y responder preguntas relacionadas con:

- Tiempos y costos de envío.
- Seguimiento e incidencias logísticas.
- Garantías de productos.
- Reembolsos y devoluciones.
- Métodos de pago.
- Programa de afiliados.

Las respuestas deberán generarse únicamente a partir del contenido de los documentos proporcionados e indicar el archivo y la página utilizados como fuente.

## Estado del proyecto

**Fase actual: configuración inicial completada.**

La aplicación ya permite validar:

- La carga de variables de entorno.
- La configuración local de Gemini.
- La detección de los cinco documentos PDF.
- La ejecución de la interfaz con Streamlit.
- La instalación de dependencias en Python 3.11.
- La preparación del proyecto para Docker.
- La ruta destinada al índice vectorial FAISS.

Actualmente, la interfaz muestra el estado de los componentes del proyecto, pero la recuperación semántica y la generación de respuestas todavía no han sido implementadas.

### Estado de los componentes

| Componente | Estado |
|---|---|
| Entorno Python 3.11 | Configurado |
| Variables de entorno | Configuradas localmente |
| Documentos PDF | 5 documentos detectados |
| Interfaz Streamlit | Funcionando |
| Lectura y fragmentación de PDF | Pendiente |
| Embeddings con Gemini | Pendiente |
| Índice FAISS | Pendiente |
| Recuperación semántica | Pendiente |
| Cadena RAG | Pendiente |
| Agente con LangGraph | Pendiente |
| Despliegue en OCI | Pendiente |

## Tecnologías

### Configuradas actualmente

- Python 3.11.
- Streamlit.
- python-dotenv.
- Pydantic.
- Docker.
- Pytest.

### Utilizadas en las siguientes etapas

- LangChain.
- LangGraph.
- Google Gemini.
- Gemini Embeddings.
- PyMuPDF.
- FAISS.
- Oracle Cloud Infrastructure.

## Arquitectura

El proyecto utiliza una arquitectura modular simplificada, organizada en las siguientes capas:

| Capa | Responsabilidad |
|---|---|
| `core` | Configuración general y variables de entorno |
| `domain` | Modelos y estructuras de datos |
| `application` | Casos de uso de indexación, RAG y agente |
| `infrastructure` | Integración con PDF, Gemini y FAISS |
| `presentation` | Interfaz de usuario con Streamlit |

### Flujo de indexación

La indexación se ejecutará cuando los documentos sean agregados o modificados.

```text
Documentos PDF
      ↓
Extracción de texto
      ↓
Normalización de metadatos
      ↓
Fragmentación del contenido
      ↓
Generación de embeddings
      ↓
Creación del índice FAISS
```

### Flujo de consulta

```text
Pregunta del usuario
      ↓
Embedding de la pregunta
      ↓
Búsqueda en FAISS
      ↓
Recuperación de fragmentos relevantes
      ↓
Contexto + pregunta
      ↓
Gemini
      ↓
Respuesta con documentos y páginas
```

## Corpus documental

Los documentos se encuentran en:

```text
data/documents/
```

El corpus inicial está compuesto por:

1. Guía de Tiempos y Costos de Envío.
2. Manual de Garantía de Productos.
3. Política de Reembolsos y Devoluciones.
4. Preguntas Frecuentes sobre Métodos de Pago.
5. Programa de Afiliados.

## Estructura del proyecto

```text
bimbam-buy-ai-agent/
│
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py
│
├── data/
│   ├── documents/
│   │   ├── guia_tiempos_costos_envio.pdf
│   │   ├── manual_garantia_productos.pdf
│   │   ├── politica_reembolsos_devoluciones.pdf
│   │   ├── preguntas_metodos_pago.pdf
│   │   └── programa_afiliados.pdf
│   │
│   └── evaluation/
│       └── questions.json
│
├── notebooks/
│   └── Curso_Agentes_de_IA_y_RAG.ipynb
│
├── scripts/
│   └── index_documents.py
│
├── storage/
│   └── .gitkeep
│
├── bimbam_assistant/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   ├── application/
│   │   ├── __init__.py
│   │   ├── indexing_service.py
│   │   ├── rag_service.py
│   │   └── agent_service.py
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   ├── gemini_provider.py
│   │   └── faiss_store.py
│   │
│   └── presentation/
│       ├── __init__.py
│       └── streamlit_app.py
│
└── tests/
    ├── test_pdf_loader.py
    ├── test_retrieval.py
    └── test_rag_service.py
```

## Requisitos previos

Para ejecutar el proyecto localmente se necesita:

- Git.
- Python 3.11.
- Conda o `venv`.
- Una clave de Google Gemini API.
- Docker para la ejecución contenerizada.

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/JSarabino/bimbam-buy-ai-agent.git
cd bimbam-buy-ai-agent
```

### 2. Crear el entorno virtual

#### Windows con Conda

```powershell
conda create --prefix .\.venv python=3.11 -y
conda activate .\.venv
```

#### Windows con Python `venv`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux o macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verificar la versión activa:

```bash
python --version
```

El resultado debe corresponder a Python 3.11.

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprobar la instalación:

```bash
python -m pip check
```

### 4. Configurar las variables de entorno

Crear `.env` a partir de la plantilla.

#### Windows

```powershell
Copy-Item .env.example .env
```

#### Linux o macOS

```bash
cp .env.example .env
```

Después, abrir `.env` y agregar la clave de Gemini:

```env
GOOGLE_API_KEY=tu_clave
```

El archivo `.env` es privado y está excluido de Git mediante `.gitignore`.

### 5. Agregar los documentos

Los documentos deben estar ubicados en:

```text
data/documents/
```

### 6. Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

## Ejecución con Docker

### Construir la imagen

```bash
docker build -t bimbam-buy-ai-agent .
```

### Ejecutar el contenedor

#### PowerShell

```powershell
docker run --rm --name bimbam-assistant --env-file .env -p 8501:8501 bimbam-buy-ai-agent
```

#### Linux o macOS

```bash
docker run \
  --rm \
  --name bimbam-assistant \
  --env-file .env \
  -p 8501:8501 \
  bimbam-buy-ai-agent
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

## Variables de entorno

Las principales variables del proyecto son:

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | Clave privada de Google Gemini |
| `GEMINI_CHAT_MODEL` | Modelo utilizado para generar respuestas |
| `GEMINI_EMBEDDING_MODEL` | Modelo utilizado para embeddings |
| `DOCUMENTS_PATH` | Ruta de los documentos PDF |
| `FAISS_INDEX_PATH` | Ruta del índice vectorial |
| `CHUNK_SIZE` | Tamaño de los fragmentos |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos |
| `RETRIEVAL_K` | Cantidad máxima de fragmentos recuperados |
| `RETRIEVAL_SCORE_THRESHOLD` | Umbral mínimo de similitud |

Los nombres y valores de ejemplo se encuentran en `.env.example`.

## Seguridad

El proyecto aplica las siguientes medidas:

- La clave de Gemini se almacena únicamente en `.env`.
- `.env` está excluido mediante `.gitignore`.
- `.env` también está excluido mediante `.dockerignore`.
- Docker recibe las variables durante la ejecución.
- La aplicación no muestra la clave en la interfaz.
- El contenedor se ejecuta con un usuario sin privilegios.
- El entorno virtual `.venv` no se almacena en GitHub.
- El índice FAISS generado tampoco se almacena en el repositorio.

## Índice vectorial

El índice será almacenado en:

```text
storage/faiss_index/
```

Este directorio se generará automáticamente y contendrá archivos como:

```text
storage/faiss_index/
├── index.faiss
└── index.pkl
```

El índice no se subirá a GitHub porque puede reconstruirse a partir de los documentos originales.

El proceso será ejecutado mediante:

```bash
python scripts/index_documents.py
```

Esta funcionalidad todavía está pendiente de implementación.

## Pruebas

Las pruebas estarán organizadas en:

```text
tests/
```

Se contemplan pruebas para:

- Lectura de documentos.
- Conservación de metadatos.
- Fragmentación del contenido.
- Recuperación de información.
- Generación de respuestas.
- Consultas sin información suficiente.

Cuando estén implementadas podrán ejecutarse con:

```bash
python -m pytest
```

## Próximas etapas

1. Implementar la configuración central del proyecto.
2. Implementar la lectura de los cinco documentos PDF.
3. Normalizar los metadatos de documento y página.
4. Dividir el contenido en fragmentos.
5. Crear embeddings con Gemini.
6. Generar y persistir el índice FAISS.
7. Implementar la recuperación semántica.
8. Construir la cadena RAG.
9. Mostrar respuestas con documentos y páginas.
10. Implementar el triaje y el agente con LangGraph.
11. Crear el banco de preguntas de evaluación.
12. Construir y probar la imagen Docker.
13. Desplegar la aplicación en OCI Compute.

## Alcance actual

La versión `0.1.0` corresponde a la configuración inicial del proyecto.

En esta versión:

- La interfaz de Streamlit funciona localmente.
- Las variables de entorno son detectadas.
- Los cinco documentos PDF son reconocidos.
- La clave de Gemini puede configurarse de manera segura.
- El índice FAISS todavía está pendiente.
- El campo de consulta permanece deshabilitado hasta implementar el servicio RAG.

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.