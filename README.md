# BimBam Assistant

Agente de inteligencia artificial basado en **RAG** (*Retrieval-Augmented Generation*) para consultar las políticas y documentos corporativos de BimBam Buy mediante preguntas en lenguaje natural.

## Descripción

BimBam Buy es un e-commerce multiplataforma enfocado en ofrecer una experiencia de compra digital ágil, segura y orientada al cliente.

El objetivo del proyecto es construir un asistente que responda consultas relacionadas con:

- Tiempos y costos de envío.
- Seguimiento e incidencias logísticas.
- Garantías de productos.
- Reembolsos y devoluciones.
- Métodos de pago.
- Programa de afiliados.

Las respuestas deberán generarse exclusivamente a partir del corpus documental e indicar el documento y la página utilizados como fuente.

## Estado del proyecto

**Fase actual: indexación vectorial completada.**

El proyecto ya permite:

- Cargar y validar la configuración desde variables de entorno.
- Detectar los cinco documentos PDF.
- Extraer el texto página por página con PyMuPDF.
- Aplicar una limpieza conservadora al contenido.
- Normalizar identificadores, nombres y categorías documentales.
- Conservar metadatos de archivo y página para futuras citas.
- Detectar páginas vacías o candidatas a OCR.
- Dividir el contenido mediante `RecursiveCharacterTextSplitter`.
- Generar identificadores únicos para cada chunk.
- Crear embeddings con `gemini-embedding-001`.
- Procesar los embeddings por lotes con pausas y reintentos ante límites de uso.
- Construir un índice FAISS con similitud coseno.
- Persistir y volver a cargar el índice vectorial.
- Validar el pipeline mediante `scripts/index_documents.py`.
- Consultar el estado del corpus y del índice desde Streamlit.

La recuperación semántica como servicio y la generación de respuestas con un modelo de chat todavía están pendientes.

### Resultado actual del corpus

| Métrica | Resultado |
|---|---:|
| Documentos PDF | 5 |
| Páginas procesadas | 57 |
| Chunks generados | 108 |
| Vectores almacenados | 108 |
| Dimensión de cada embedding | 3072 |
| Tamaño máximo observado del chunk | 999 caracteres |
| Categorías reconocidas | 5 |
| Chunks sin clasificar | 0 |
| Identificadores duplicados | 0 |

Estos valores corresponden al corpus actual y pueden cambiar cuando se agreguen o modifiquen documentos.

### Estado de los componentes

| Componente | Estado |
|---|---|
| Entorno Python 3.11 | Completado |
| Configuración central | Completada |
| Documentos PDF | 5 detectados |
| Extracción y limpieza | Completadas |
| Metadatos y categorías | Completados |
| Fragmentación en chunks | Completada |
| Embeddings con Gemini | Completados |
| Índice FAISS | Completado |
| Persistencia del índice | Completada |
| Interfaz Streamlit | Actualizada |
| Recuperación semántica | Pendiente |
| Cadena RAG | Pendiente |
| Agente con LangGraph | Pendiente |
| Pruebas automatizadas | Pendientes |
| Despliegue en OCI | Pendiente |

## Tecnologías

### Utilizadas actualmente

- Python 3.11.
- LangChain Core.
- LangChain Text Splitters.
- Google Gemini Embeddings.
- PyMuPDF.
- FAISS CPU.
- NumPy.
- Streamlit.
- python-dotenv.
- Pydantic.
- Pytest.
- Docker.

### Preparadas para las siguientes etapas

- Google Gemini como modelo de chat.
- LangGraph.
- Oracle Cloud Infrastructure.

## Arquitectura

El proyecto utiliza una arquitectura modular simplificada:

| Capa | Responsabilidad |
|---|---|
| `core` | Configuración general, variables de entorno y rutas |
| `domain` | Modelos y estructuras de datos |
| `application` | Casos de uso de fragmentación, indexación, RAG y agente |
| `infrastructure` | Integraciones con PDF, Gemini y FAISS |
| `presentation` | Interfaz de usuario con Streamlit |

### Flujo de indexación

```text
Documentos PDF
      ↓
Extracción por página                 ✓
      ↓
Limpieza conservadora                 ✓
      ↓
Normalización de metadatos            ✓
      ↓
Fragmentación del contenido           ✓
      ↓
Generación de embeddings              ✓
      ↓
Normalización de vectores             ✓
      ↓
Creación y persistencia en FAISS      ✓
```

### Flujo de consulta

```text
Pregunta del usuario
      ↓
Embedding de la pregunta              ✓ como función de infraestructura
      ↓
Búsqueda en FAISS                     ✓ como función de infraestructura
      ↓
Recuperación y filtrado               Pendiente como servicio
      ↓
Contexto + pregunta                   Pendiente
      ↓
Gemini
      ↓
Respuesta con documento y página
```

## Corpus documental

Los documentos se encuentran en:

```text
data/documents/
```

El corpus inicial está compuesto por:

1. Guía de Tiempos y Costos de Envío de BimBam Buy.
2. Manual de Garantía de Productos de BimBam Buy.
3. Política de Reembolsos y Devoluciones de BimBam.
4. Preguntas Frecuentes sobre Métodos de Pago de BimBam Buy.
5. Programa de Afiliados de BimBam Buy.

Cada página conserva metadatos como:

- Identificador y nombre del documento.
- Categoría.
- Nombre y ruta del archivo.
- Número e índice de página.
- Total de páginas.
- Método de extracción.
- Cantidad de caracteres.
- Estado de página vacía.
- Posible necesidad de OCR.

Cada chunk agrega:

- `chunk_id`.
- `chunk_index`.
- `chunk_number`.
- Cantidad de caracteres.
- Tamaño y solapamiento configurados.

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
│   ├── .gitkeep
│   └── faiss_index/                 # Generado localmente e ignorado por Git
│       ├── index.faiss
│       ├── documents.json
│       └── manifest.json
│
├── bimbam_assistant/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── indexing_service.py
│   │   ├── rag_service.py
│   │   └── agent_service.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   ├── gemini_provider.py
│   │   └── faiss_store.py
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

#### Windows con `venv`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux o macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verificar la versión:

```bash
python --version
```

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Configurar las variables de entorno

#### Windows

```powershell
Copy-Item .env.example .env
```

#### Linux o macOS

```bash
cp .env.example .env
```

Agregar la clave de Gemini en `.env`:

```env
GOOGLE_API_KEY=tu_clave
```

El archivo `.env` es privado y no debe subirse al repositorio.

### 5. Agregar los documentos

Copiar los PDF en:

```text
data/documents/
```

## Procesamiento e indexación

Ejecutar:

```bash
python scripts/index_documents.py
```

El script:

1. Busca los PDF configurados.
2. Extrae el texto página por página.
3. Limpia y normaliza el contenido.
4. Asigna metadatos y categorías.
5. Divide las páginas en chunks.
6. Valida identificadores, fuentes, páginas, tamaños y categorías.
7. Genera los embeddings con Gemini por lotes.
8. Aplica pausas y reintentos cuando la API alcanza un límite temporal.
9. Construye el índice `IndexFlatIP` con vectores normalizados.
10. Guarda el índice y su información asociada.
11. Devuelve código de salida `0` cuando la indexación termina correctamente.

En PowerShell:

```powershell
python scripts/index_documents.py
$LASTEXITCODE
```

El proceso genera:

```text
storage/faiss_index/
├── index.faiss
├── documents.json
└── manifest.json
```

- `index.faiss`: vectores y estructura de búsqueda.
- `documents.json`: texto y metadatos asociados a cada vector.
- `manifest.json`: modelo, dimensión, métrica y configuración del índice.

El directorio generado está ignorado por Git. Después de clonar el repositorio es necesario reconstruirlo mediante el script.

## Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

La página inicial muestra:

- Estado de la clave de Gemini.
- Número de documentos.
- Páginas procesadas.
- Chunks generados.
- Páginas vacías y candidatas a OCR.
- Categorías y resumen por documento.
- Disponibilidad y validación del índice FAISS.
- Modelo de embeddings.
- Dimensión y cantidad de vectores.
- Tipo de índice y métrica de similitud.

El campo de consulta permanece deshabilitado hasta implementar el servicio de recuperación semántica y la cadena RAG.

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

La estrategia para construir o montar el índice dentro del despliegue contenerizado se completará junto con la etapa de despliegue.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | Clave privada de Google Gemini |
| `GEMINI_CHAT_MODEL` | Modelo utilizado para generar respuestas |
| `GEMINI_EMBEDDING_MODEL` | Modelo utilizado para embeddings |
| `DOCUMENTS_PATH` | Ruta de los documentos PDF |
| `FAISS_INDEX_PATH` | Ruta del índice vectorial |
| `CHUNK_SIZE` | Tamaño máximo de los fragmentos |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos |
| `RETRIEVAL_K` | Cantidad máxima de fragmentos recuperados |
| `RETRIEVAL_SCORE_THRESHOLD` | Umbral mínimo de similitud |

## Seguridad

- La clave de Gemini se almacena únicamente en `.env`.
- `.env` está excluido mediante `.gitignore` y `.dockerignore`.
- Docker recibe las variables durante la ejecución.
- La aplicación nunca muestra la clave.
- El contenedor se ejecuta con un usuario sin privilegios.
- `.venv` no se almacena en GitHub.
- `storage/faiss_index/` no se almacena en GitHub.
- El índice se carga únicamente desde archivos generados por la aplicación.

## Índice vectorial

La implementación actual utiliza:

```text
Modelo              gemini-embedding-001
Dimensión           3072
Índice              IndexFlatIP
Métrica             similitud coseno
Vectores            108
```

Los embeddings se normalizan antes de agregarlos al índice. Así, el producto interno calculado por `IndexFlatIP` representa la similitud coseno.

El manifiesto permite verificar que el índice cargado sea compatible con el modelo configurado y con la dimensión esperada.

## Pruebas

Las pruebas estarán organizadas en:

```text
tests/
```

Se contemplan pruebas para:

- Lectura y limpieza de documentos.
- Conservación de metadatos.
- Fragmentación del contenido.
- Creación, persistencia y carga del índice.
- Recuperación semántica.
- Generación de respuestas.
- Consultas sin información suficiente.

Cuando estén implementadas:

```bash
python -m pytest
```

## Próximas etapas

1. Implementar el servicio de recuperación semántica.
2. Aplicar `RETRIEVAL_K` y `RETRIEVAL_SCORE_THRESHOLD`.
3. Devolver fragmentos con documento, página y puntuación.
4. Construir la cadena RAG.
5. Mostrar respuestas y fuentes en Streamlit.
6. Implementar el triaje y el agente con LangGraph.
7. Crear el banco de preguntas de evaluación.
8. Implementar las pruebas automatizadas.
9. Construir y probar la imagen Docker con su estrategia de índice.
10. Desplegar la aplicación en OCI Compute.

## Alcance actual

El hito actual cubre:

```text
PDF
  → extracción
  → limpieza
  → metadatos
  → chunks validados
  → embeddings
  → índice FAISS persistido y validado
```

La aplicación todavía no genera respuestas ni expone la recuperación semántica al usuario.

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.
