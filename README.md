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

Las respuestas deberán generarse exclusivamente a partir del corpus documental y mostrar el documento y la página utilizados como fuente.

## Estado del proyecto

**Fase actual: procesamiento documental y fragmentación completados.**

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
- Validar el procesamiento mediante `scripts/index_documents.py`.
- Consultar el estado del procesamiento desde Streamlit.

### Resultado actual del corpus

| Métrica | Resultado |
|---|---:|
| Documentos PDF | 5 |
| Páginas procesadas | 57 |
| Chunks generados | 108 |
| Tamaño máximo observado | 999 caracteres |
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
| Extracción de texto | Completada |
| Limpieza del contenido | Completada |
| Metadatos y categorías | Completados |
| Fragmentación en chunks | Completada |
| Validación del procesamiento | Completada |
| Interfaz Streamlit | Actualizada |
| Embeddings con Gemini | Pendiente |
| Índice FAISS | Pendiente |
| Recuperación semántica | Pendiente |
| Cadena RAG | Pendiente |
| Agente con LangGraph | Pendiente |
| Despliegue en OCI | Pendiente |

## Tecnologías

### Utilizadas actualmente

- Python 3.11.
- LangChain Core.
- LangChain Text Splitters.
- PyMuPDF.
- Streamlit.
- python-dotenv.
- Pydantic.
- Pytest.
- Docker.

### Preparadas para las siguientes etapas

- LangGraph.
- Google Gemini.
- Gemini Embeddings.
- FAISS.
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
Generación de embeddings              Pendiente
      ↓
Creación y persistencia en FAISS      Pendiente
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

Cada página procesada conserva metadatos como:

- Identificador y nombre del documento.
- Categoría.
- Nombre y ruta del archivo.
- Número e índice de página.
- Total de páginas.
- Método de extracción.
- Cantidad de caracteres.
- Estado de página vacía.
- Posible necesidad de OCR.

Cada chunk agrega además:

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
│   └── .gitkeep
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

## Procesamiento documental

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
6. Verifica identificadores, fuentes, páginas, tamaños y categorías.
7. Devuelve código de salida `0` cuando el procesamiento es válido.

En PowerShell se puede verificar el código de salida con:

```powershell
python scripts/index_documents.py
$LASTEXITCODE
```

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
- Estado del índice FAISS.

El campo de consulta permanece deshabilitado hasta implementar el servicio RAG.

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
- El índice FAISS generado tampoco se almacenará en el repositorio.

## Índice vectorial

El índice se almacenará en:

```text
storage/faiss_index/
```

En la siguiente etapa, `scripts/index_documents.py` se ampliará para generar:

```text
storage/faiss_index/
├── index.faiss
└── index.pkl
```

Actualmente el script finaliza después de validar los chunks; todavía no consume la API de Gemini ni genera el índice.

## Pruebas

Las pruebas estarán organizadas en:

```text
tests/
```

Se contemplan pruebas para:

- Lectura y limpieza de documentos.
- Conservación de metadatos.
- Fragmentación del contenido.
- Recuperación semántica.
- Generación de respuestas.
- Consultas sin información suficiente.

Cuando estén implementadas:

```bash
python -m pytest
```

## Próximas etapas

1. Implementar el proveedor de embeddings con Gemini.
2. Generar embeddings para los chunks.
3. Crear y persistir el índice FAISS.
4. Implementar la recuperación semántica.
5. Construir la cadena RAG.
6. Mostrar respuestas con documento y página.
7. Implementar el triaje y el agente con LangGraph.
8. Crear el banco de preguntas de evaluación.
9. Implementar las pruebas automatizadas.
10. Construir y probar la imagen Docker.
11. Desplegar la aplicación en OCI Compute.

## Alcance actual

El hito actual cubre la preparación documental previa a los embeddings:

```text
PDF → extracción → limpieza → metadatos → chunks validados
```

La aplicación todavía no genera respuestas ni realiza búsquedas semánticas.

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.
