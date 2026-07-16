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

**Fase actual: recuperación semántica completada.**

El proyecto ya permite:

- Cargar y validar la configuración desde variables de entorno.
- Detectar los cinco documentos PDF.
- Extraer el texto página por página con PyMuPDF.
- Limpiar y clasificar el contenido.
- Conservar metadatos de documento, categoría y página.
- Dividir el corpus en chunks trazables.
- Crear embeddings con `gemini-embedding-001`.
- Procesar los embeddings por lotes con pausas y reintentos.
- Construir y persistir un índice FAISS con similitud coseno.
- Transformar una pregunta en un embedding.
- Buscar los chunks más próximos semánticamente.
- Aplicar `top k`, umbral mínimo de similitud y filtros por metadatos.
- Devolver documento, página, categoría, puntuación y texto.
- Ensamblar el contexto que utilizará la futura cadena RAG.
- Ejecutar consultas semánticas desde Streamlit.

La generación de la respuesta final con un modelo de chat y el agente con LangGraph todavía están pendientes.

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
| Resultados recuperados por defecto | 4 |
| Umbral mínimo configurado | 0.30 |

Estos valores corresponden al corpus y a la configuración actuales.

### Estado de los componentes

| Componente | Estado |
|---|---|
| Entorno Python 3.11 | Completado |
| Configuración central | Completada |
| Extracción y limpieza | Completadas |
| Metadatos y categorías | Completados |
| Fragmentación en chunks | Completada |
| Embeddings con Gemini | Completados |
| Índice FAISS | Completado |
| Persistencia y carga del índice | Completadas |
| Embedding de consultas | Completado |
| Búsqueda semántica | Completada |
| Filtros por metadatos | Completados |
| Ensamblaje del contexto | Completado |
| Interfaz de recuperación en Streamlit | Completada |
| Reranking | No implementado en el baseline |
| Generación de respuestas RAG | Pendiente |
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
| `domain` | Modelos de recuperación y estructuras de datos |
| `application` | Casos de uso de fragmentación, indexación, recuperación, RAG y agente |
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
Normalización de la consulta           ✓
      ↓
Embedding de la pregunta               ✓
      ↓
Búsqueda exacta en FAISS               ✓
      ↓
Top k + umbral + filtros               ✓
      ↓
Fragmentos con documento y página      ✓
      ↓
Ensamblaje del contexto                ✓
      ↓
Generación de la respuesta con Gemini  Pendiente
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

El directorio generado está ignorado por Git. Después de clonar el repositorio es necesario reconstruirlo mediante el script.

## Recuperación semántica

La función principal está en:

```text
bimbam_assistant/application/rag_service.py
```

Ejemplo básico:

```python
from bimbam_assistant.application.rag_service import retrieve_documents

response = retrieve_documents(
    "¿Cuánto tarda un reembolso?"
)

for result in response.results:
    print(
        result.rank,
        result.score,
        result.metadata["document_name"],
        result.metadata["page_number"],
    )
```

La recuperación utiliza por defecto:

```text
RETRIEVAL_K=4
RETRIEVAL_SCORE_THRESHOLD=0.30
```

También admite filtros por metadatos:

```python
response = retrieve_documents(
    "¿Qué debo presentar para solicitar una garantía?",
    filters={
        "category": "garantias",
    },
)
```

El resultado estructurado contiene:

- Consulta normalizada.
- Lista ordenada de fragmentos.
- Identificador del vector.
- Puntuación de similitud.
- Texto recuperado.
- Documento, página y categoría.
- Filtros aplicados.
- Contexto ensamblado para el futuro LLM.

Cada consulta genera únicamente un embedding nuevo. Los 108 embeddings documentales ya están almacenados y la búsqueda se ejecuta localmente con FAISS.

## Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación estará disponible en:

```text
http://localhost:8501
```

La interfaz muestra:

- Estado del procesamiento documental.
- Disponibilidad y validación del índice FAISS.
- Modelo, dimensión, tipo de índice y métrica.
- Formulario de consulta semántica.
- Filtro opcional por categoría.
- Tabla de resultados ordenados.
- Documento, página, categoría y puntuación de cada fragmento.
- Texto recuperado.
- Contexto completo ensamblado.

El formulario se habilita cuando:

1. El índice FAISS está disponible y es válido.
2. `GOOGLE_API_KEY` está configurada.

La interfaz todavía no genera la respuesta final. Por ahora muestra la evidencia recuperada que se entregará al modelo generativo.

## Reranking

El baseline actual utiliza búsqueda exacta con `IndexFlatIP`, `top k`, umbral y filtros.

El reranking se deja pendiente hasta evaluar sistemáticamente la calidad de recuperación. Si el baseline muestra resultados irrelevantes o redundantes, se podrá implementar el flujo:

```text
FAISS recupera más candidatos
          ↓
Reranker reordena por relevancia
          ↓
Se conservan los mejores fragmentos
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

## Pruebas manuales realizadas

La recuperación fue verificada con:

- Consulta sobre el plazo de reembolsos.
- Consulta de garantías filtrada por categoría.
- Consulta fuera del dominio con un umbral de `0.95`.
- Contexto vacío cuando no hay resultados.
- Conservación de documento, página, categoría y puntuación.

Las pruebas automatizadas se implementarán en `tests/`.

## Próximas etapas

1. Construir la cadena RAG con el modelo de chat de Gemini.
2. Diseñar el prompt con restricciones de fidelidad al contexto.
3. Generar respuestas con documento y página.
4. Manejar consultas sin evidencia suficiente.
5. Evaluar sistemáticamente la recuperación.
6. Decidir si es necesario incorporar reranking.
7. Implementar el triaje y el agente con LangGraph.
8. Crear el banco de preguntas de evaluación.
9. Implementar las pruebas automatizadas.
10. Construir y probar la imagen Docker.
11. Desplegar la aplicación en OCI Compute.

## Alcance actual

El hito actual cubre:

```text
PDF
  → extracción
  → limpieza
  → metadatos
  → chunks validados
  → embeddings
  → índice FAISS
  → embedding de consulta
  → búsqueda semántica
  → filtros y umbral
  → contexto ensamblado
```

La aplicación todavía no genera la respuesta final con el modelo de chat.

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.
