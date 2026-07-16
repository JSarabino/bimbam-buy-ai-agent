# BimBam Assistant

Agente de inteligencia artificial basado en **RAG** (*Retrieval-Augmented Generation*) para consultar las políticas y documentos corporativos de BimBam Buy mediante preguntas en lenguaje natural.

## Descripción

BimBam Assistant recupera información desde un corpus corporativo, genera una respuesta con Gemini y ejecuta una verificación independiente antes de mostrarla.

El asistente cubre consultas sobre:

- Tiempos y costos de envío.
- Seguimiento e incidencias logísticas.
- Garantías de productos.
- Reembolsos y devoluciones.
- Métodos de pago.
- Programa de afiliados.

Las respuestas se restringen al contexto documental recuperado y conservan trazabilidad mediante referencias como `[Fuente 1]`, junto con el documento y la página de origen.

## Estado del proyecto

**Fase actual: cadena RAG con verificación automática completada.**

El proyecto ya permite:

- Procesar cinco documentos PDF.
- Extraer y limpiar el texto página por página.
- Conservar metadatos de documento, categoría y página.
- Generar 108 chunks trazables.
- Crear embeddings con Gemini.
- Construir y persistir un índice FAISS.
- Transformar preguntas en embeddings.
- Aplicar búsqueda semántica, `top k`, umbral y filtros.
- Ensamblar el contexto documental.
- Generar respuestas con Gemini.
- Comprobar automáticamente citas y respaldo semántico.
- Rechazar respuestas no sustentadas.
- Evitar la generación cuando no existe evidencia suficiente.
- Mostrar respuesta, verificación, fuentes y contexto en Streamlit.
- Ofrecer contactos sintéticos de demostración únicamente durante un fallback.

### Resultado actual

| Métrica | Resultado |
|---|---:|
| Documentos PDF | 5 |
| Páginas procesadas | 57 |
| Chunks generados | 108 |
| Vectores almacenados | 108 |
| Dimensión de cada embedding | 3072 |
| Categorías reconocidas | 5 |
| Resultados recuperados por defecto | 4 |
| Umbral mínimo configurado | 0.30 |
| Confianza mínima de verificación | 0.75 |

### Estado de los componentes

| Componente | Estado |
|---|---|
| Extracción y limpieza | Completadas |
| Fragmentación y metadatos | Completados |
| Embeddings de documentos | Completados |
| Índice FAISS | Completado |
| Embedding de consultas | Completado |
| Recuperación semántica | Completada |
| Filtros y umbral | Completados |
| Ensamblaje del contexto | Completado |
| Generación RAG | Completada |
| Citas `[Fuente N]` | Implementadas |
| Validación determinística de citas | Implementada |
| Verificación semántica estructurada | Implementada |
| Rechazo de respuestas no respaldadas | Implementado |
| Fallback sin evidencia | Implementado |
| Contactos sintéticos de demostración | Implementados |
| Streamlit | Actualizado |
| Reranking | Pendiente de evaluación |
| Banco de evaluación | Pendiente |
| Agente con LangGraph | Pendiente |
| Despliegue en OCI | Pendiente |

## Tecnologías

- Python 3.11.
- LangChain Core.
- LangChain Google GenAI.
- LangChain Text Splitters.
- Google Gemini.
- PyMuPDF.
- FAISS CPU.
- NumPy.
- Streamlit.
- python-dotenv.
- Pydantic.
- Pytest.
- Docker.

## Arquitectura

| Capa | Responsabilidad |
|---|---|
| `core` | Configuración, variables de entorno y rutas |
| `domain` | Modelos de recuperación, respuesta, verificación y contactos |
| `application` | Indexación, recuperación, generación y verificación |
| `infrastructure` | Integraciones con PDF, Gemini y FAISS |
| `presentation` | Interfaz Streamlit |

### Flujo completo

```text
PDF
 ↓
Extracción y limpieza
 ↓
Chunks con metadatos
 ↓
Embeddings
 ↓
FAISS
 ↓
Pregunta del usuario
 ↓
Embedding de consulta
 ↓
Top k + umbral + filtros
 ↓
Contexto documental
 ↓
Gemini genera respuesta
 ↓
Validación de citas
 ↓
Gemini verifica respaldo semántico
 ↓
Respuesta verificada o fallback seguro
```

## Corpus documental

Los documentos están en:

```text
data/documents/
```

El corpus contiene:

1. Guía de Tiempos y Costos de Envío de BimBam Buy.
2. Manual de Garantía de Productos de BimBam Buy.
3. Política de Reembolsos y Devoluciones de BimBam.
4. Preguntas Frecuentes sobre Métodos de Pago de BimBam Buy.
5. Programa de Afiliados de BimBam Buy.

Los documentos mencionan áreas y canales generales de atención, pero no contienen direcciones de correo, teléfonos o URL concretas. Por esa razón, los contactos mostrados por el fallback son datos sintéticos de demostración y no forman parte de la base de conocimiento.

## Estructura del proyecto

```text
bimbam-buy-ai-agent/
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py
├── data/
│   ├── documents/
│   └── evaluation/
│       └── questions.json
├── notebooks/
│   └── Curso_Agentes_de_IA_y_RAG.ipynb
├── scripts/
│   └── index_documents.py
├── storage/
│   ├── .gitkeep
│   └── faiss_index/
│       ├── index.faiss
│       ├── documents.json
│       └── manifest.json
├── bimbam_assistant/
│   ├── core/
│   │   └── config.py
│   ├── domain/
│   │   ├── models.py
│   │   └── support_contacts.py
│   ├── application/
│   │   ├── indexing_service.py
│   │   ├── rag_service.py
│   │   ├── verification_service.py
│   │   └── agent_service.py
│   ├── infrastructure/
│   │   ├── pdf_loader.py
│   │   ├── gemini_provider.py
│   │   └── faiss_store.py
│   └── presentation/
│       └── streamlit_app.py
└── tests/
    ├── test_pdf_loader.py
    ├── test_retrieval.py
    └── test_rag_service.py
```

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/JSarabino/bimbam-buy-ai-agent.git
cd bimbam-buy-ai-agent
```

### 2. Crear el entorno

```powershell
conda create --prefix .\.venv python=3.11 -y
conda activate .\.venv
```

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Configurar variables

```powershell
Copy-Item .env.example .env
```

Agregar la clave:

```env
GOOGLE_API_KEY=tu_clave
```

### 5. Generar el índice

```bash
python scripts/index_documents.py
```

El índice se guarda localmente en:

```text
storage/faiss_index/
├── index.faiss
├── documents.json
└── manifest.json
```

Este directorio está ignorado por Git y debe reconstruirse después de clonar el repositorio.

## Generación RAG

La función principal es:

```python
from bimbam_assistant.application.rag_service import answer_question

response = answer_question(
    "¿Cuánto tarda un reembolso?"
)

print(response.answer)
print(response.verification.status)
print(response.verification.confidence)
```

`RagResponse` contiene:

- Pregunta normalizada.
- Respuesta final.
- Recuperación utilizada.
- Modelo generativo.
- Indicador de uso del contexto.
- Verificación automática.
- Fuentes documentales.
- Contacto sintético opcional para el fallback.

## Verificación automática

La verificación aplica dos controles.

### 1. Validación determinística

Se extraen las referencias con formato:

```text
[Fuente 1]
[Fuente 2]
```

Después se comprueba que:

- Exista al menos una cita cuando hay evidencia.
- Cada número citado corresponda a una fuente recuperada.
- No aparezcan referencias inexistentes como `[Fuente 99]`.

### 2. Verificación semántica

Un segundo llamado estructurado a Gemini compara:

- La pregunta.
- La respuesta generada.
- El contexto documental autorizado.

El verificador devuelve un modelo Pydantic con:

- `status`: `verified`, `rejected` o `not_applicable`.
- `passed`.
- `semantic_supported`.
- `confidence`.
- `cited_sources`.
- `invalid_citations`.
- `unsupported_claims`.
- `explanation`.

La respuesta se acepta únicamente cuando:

```text
contenido respaldado
AND confianza >= 0.75
AND existen citas
AND no hay citas inválidas
```

Cuando la validación falla, la respuesta original no se muestra y se reemplaza por un mensaje seguro.

## Fallback

### Sin evidencia

Cuando ningún fragmento supera el umbral:

- No se invoca el modelo generativo.
- No se invoca el verificador.
- Se devuelve un mensaje explícito de falta de información.
- El estado de verificación es `not_applicable`.

### Respuesta rechazada

Cuando el modelo genera contenido sin respaldo suficiente:

- El verificador marca `rejected`.
- La respuesta generada se descarta.
- Se muestra un fallback seguro.
- Se conservan los detalles técnicos para auditoría.

## Contactos sintéticos

Los contactos alternativos utilizan direcciones bajo `example.com` y están definidos en:

```text
bimbam_assistant/domain/support_contacts.py
```

Ejemplo:

```text
postventa-bimbam@example.com
```

Reglas de uso:

- Son ficticios.
- Solo se muestran en fallbacks.
- No se indexan.
- No se presentan como información extraída de los documentos.
- No se citan como fuentes.
- Deben sustituirse por contactos reales antes de un uso productivo.

## Ejecutar Streamlit

```bash
python -m streamlit run app.py
```

La interfaz presenta:

- Estado del corpus y del índice.
- Pregunta y filtro por categoría.
- Respuesta final.
- Estado de verificación.
- Confianza del verificador.
- Indicador de uso de contexto.
- Modelo generativo.
- Citas válidas e inválidas.
- Afirmaciones no respaldadas.
- Explicación de la verificación.
- Documento, página, categoría y similitud de cada fuente.
- Contexto utilizado.
- Contacto ficticio solo cuando se activa un fallback.

## Consumo de API

Una consulta con evidencia normalmente utiliza:

```text
1 solicitud de embedding para la pregunta
1 solicitud para generar la respuesta
1 solicitud para verificarla
```

Una consulta sin evidencia utiliza únicamente el embedding de la pregunta, porque no se ejecutan la generación ni la verificación semántica.

## Pruebas manuales realizadas

### Respuesta válida

La consulta:

```text
¿Cuánto tarda un reembolso?
```

produjo:

```text
Estado: verified
Confianza: 1.0
Citas: [1, 2, 3]
Citas inválidas: []
Afirmaciones no respaldadas: []
```

### Respuesta falsa

La afirmación de prueba:

```text
El reembolso tarda exactamente 90 días calendario
y se paga en criptomonedas [Fuente 99].
```

fue rechazada y el verificador detectó:

- Plazo no respaldado.
- Medio de pago inventado.
- Fuente inexistente.

## Seguridad y limitaciones

- `.env` no se almacena en Git.
- La clave de Gemini no se muestra.
- El modelo recibe el corpus como contexto, no como instrucciones.
- Las respuestas no verificadas se descartan.
- Los contactos sintéticos están explícitamente marcados.
- La verificación con otro LLM reduce el riesgo, pero no constituye una garantía matemática.
- Cada verificación agrega latencia y consumo de API.
- El reranking todavía no forma parte del baseline.

## Próximas etapas

1. Crear el banco de preguntas de evaluación.
2. Implementar pruebas automáticas de recuperación, citas y fallback.
3. Medir precisión, fidelidad y tasa de rechazo.
4. Ajustar el umbral de recuperación y la confianza de verificación.
5. Decidir si se necesita reranking.
6. Implementar el triaje y el agente con LangGraph.
7. Construir y probar Docker.
8. Desplegar en OCI Compute.

## Alcance actual

```text
PDF
  → extracción
  → chunks
  → embeddings
  → FAISS
  → recuperación
  → generación
  → verificación automática
  → respuesta verificada o fallback
```

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.
