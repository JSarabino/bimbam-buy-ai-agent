# BimBam Assistant

Agente de inteligencia artificial basado en **RAG** (*Retrieval-Augmented Generation*) para consultar las políticas y documentos corporativos de BimBam Buy mediante preguntas en lenguaje natural.

## Descripción

BimBam Assistant procesa un corpus documental corporativo, recupera los fragmentos más relevantes para cada consulta, genera una respuesta con Gemini y ejecuta una verificación independiente antes de mostrarla.

El asistente cubre consultas sobre:

- Tiempos y costos de envío.
- Seguimiento e incidencias logísticas.
- Garantías de productos.
- Reembolsos y devoluciones.
- Métodos de pago.
- Programa de afiliados.

Las respuestas se restringen al contexto documental recuperado y conservan trazabilidad mediante referencias como `[Fuente 1]`, junto con el documento, la página, la categoría y el fragmento de origen.

## Estado del proyecto

**Fase actual: RAG conversacional, verificación, monitoreo, mantenimiento documental y banco de evaluación controlado completados.**

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
- Validar automáticamente las citas.
- Verificar semánticamente el respaldo de la respuesta.
- Rechazar respuestas no sustentadas.
- Evitar la generación cuando no existe evidencia suficiente.
- Mantener un historial conversacional durante la sesión.
- Interpretar preguntas de seguimiento usando las preguntas recientes.
- Mostrar fuentes y detalles técnicos de forma desplegable.
- Registrar feedback positivo o negativo por respuesta.
- Persistir interacciones, errores, tiempos y feedback en SQLite.
- Mostrar métricas de calidad y posibles vacíos de conocimiento.
- Reintentar automáticamente fallos transitorios de verificación.
- Calcular firmas SHA-256 de los documentos del corpus.
- Detectar documentos agregados, modificados, eliminados o sin cambios.
- Mantener un banco de 20 preguntas para evaluar recuperación, generación, citas y fallback.
- Validar el banco y simular el presupuesto sin consumir llamadas de Gemini.
- Ejecutar evaluaciones por lotes con un límite diario protegido.
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
| Interfaz conversacional Streamlit | Implementada |
| Historial durante la sesión | Implementado |
| Feedback por respuesta | Implementado |
| Persistencia SQLite | Implementada |
| Monitoreo de calidad | Implementado |
| Reranking | Pendiente de evaluación |
| Detección de cambios por SHA-256 | Implementada |
| Integración del detector con la indexación | Implementada |
| Omisión de indexación cuando no hay cambios | Implementada |
| Reconstrucción manual con `--force` | Implementada |
| Aviso de sincronización en Streamlit | Implementado |
| Bloqueo del chat con índice desactualizado | Implementado |
| Ejecutor programable para Windows | Implementado y probado |
| Preparación de cron para Linux/OCI | Implementada |
| Activación de cron en OCI | Pendiente del despliegue |
| Banco de evaluación de 20 preguntas | Implementado |
| Validador offline del banco | Implementado |
| Evaluador RAG con presupuesto diario | Implementado |
| Ejecución real de los lotes | Pendiente de cuota disponible |
| Agente con LangGraph | Siguiente hito |
| Despliegue en OCI | Pendiente |

## Tecnologías

- Python 3.11.
- LangChain Core.
- LangChain Google GenAI.
- LangChain Text Splitters.
- LangGraph.
- Google Gemini.
- PyMuPDF.
- FAISS CPU.
- NumPy.
- Streamlit.
- SQLite.
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
| `infrastructure` | Integraciones con PDF, Gemini, FAISS, SQLite y detección de cambios documentales |
| `presentation` | Interfaz conversacional en Streamlit |

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
 ↓
Feedback y métricas en SQLite
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
│   ├── faiss_index/
│   │   ├── index.faiss
│   │   ├── documents.json
│   │   └── manifest.json
│   └── monitoring/
│       └── bimbam_quality.db
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
│   │   ├── faiss_store.py
│   │   ├── monitoring_repository.py
│   │   └── document_change_detector.py
│   └── presentation/
│       └── streamlit_app.py
└── tests/
    ├── test_pdf_loader.py
    ├── test_retrieval.py
    ├── test_rag_service.py
    ├── test_verification_retry.py
    └── test_document_change_detector.py
```

Los directorios `storage/faiss_index/` y `storage/monitoring/` se generan localmente y no deben almacenarse en GitHub.

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/JSarabino/bimbam-buy-ai-agent.git
cd bimbam-buy-ai-agent
```

### 2. Crear el entorno

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

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Configurar variables

#### Windows

```powershell
Copy-Item .env.example .env
```

#### Linux o macOS

```bash
cp .env.example .env
```

Agregar la clave:

```env
GOOGLE_API_KEY=tu_clave
```

El archivo `.env` es privado y no debe subirse al repositorio.

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

Este directorio debe reconstruirse después de clonar el repositorio.

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

El verificador devuelve:

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

## Interfaz conversacional

Ejecutar:

```bash
python -m streamlit run app.py
```

La aplicación estará disponible normalmente en:

```text
http://localhost:8501
```

La interfaz incluye:

- Indicación explícita de que se conversa con un asistente de IA.
- Mensaje inicial con ejemplos de preguntas.
- Campo de chat.
- Filtro opcional por categoría.
- Historial de preguntas y respuestas durante la sesión.
- Continuidad básica para preguntas de seguimiento.
- Respuesta destacada visualmente.
- Feedback `👍 Útil` y `👎 No útil`.
- Fuentes, verificación y contexto en paneles desplegables.
- Botón para iniciar una conversación nueva.
- Métricas técnicas compactas.
- Panel de monitoreo de calidad.

Las últimas preguntas ayudan a interpretar seguimientos como:

```text
Usuario: ¿Cuánto tarda un reembolso?
Usuario: ¿Y si pagué en cuotas?
```

Las respuestas anteriores no se usan como evidencia. El sistema vuelve a consultar el corpus para cada pregunta.


## Reintentos automáticos de verificación

La verificación estructurada incorpora un reintento controlado para errores temporales del proveedor.

Se reintenta ante señales como:

```text
429
500
502
503
504
rate limit
resource exhausted
service unavailable
timeout
deadline exceeded
```

Configuración actual:

```python
VERIFICATION_MAX_ATTEMPTS = 2
VERIFICATION_RETRY_BASE_SECONDS = 2.0
```

Esto representa un intento inicial y un único reintento automático después de una espera de dos segundos.

No se reintenta cuando:

- La respuesta estructurada se generó correctamente.
- El verificador concluye que la respuesta no está respaldada.
- Existen citas inválidas.
- El error corresponde a una condición permanente de esquema o configuración.

Pruebas implementadas:

```bash
python -m pytest tests/test_verification_retry.py -v
```

Resultado validado:

```text
4 passed
```

## Detección de cambios documentales

El módulo:

```text
bimbam_assistant/infrastructure/document_change_detector.py
```

calcula una firma SHA-256 por cada PDF del corpus.

Cada firma incluye:

```text
relative_path
sha256
size_bytes
```

El detector clasifica los documentos como:

```text
added
modified
deleted
unchanged
```

El manifiesto esperado se almacenará en:

```text
storage/faiss_index/corpus_manifest.json
```

Ejemplo de estructura:

```json
{
  "schema_version": 1,
  "hash_algorithm": "sha256",
  "document_count": 5,
  "documents": [
    {
      "relative_path": "Manual de Garantía de Productos de BimBam Buy.pdf",
      "sha256": "firma-del-documento",
      "size_bytes": 24583
    }
  ]
}
```

La primera inspección marca los cinco PDF como agregados cuando todavía no existe un manifiesto previo.

Pruebas implementadas:

```bash
python -m pytest tests/test_document_change_detector.py -v
```

Resultado validado:

```text
6 passed
```

El detector está integrado con `scripts/index_documents.py`. La indexación se omite cuando el corpus no cambia y puede forzarse con `--force`.

El siguiente paso será:

```text
calcular hashes
→ comparar manifiestos
→ omitir indexación si no hay cambios
→ reconstruir FAISS si cambia el corpus
→ guardar el nuevo manifiesto
```

## Monitoreo persistente

La aplicación crea automáticamente:

```text
storage/monitoring/bimbam_quality.db
```

La base SQLite registra:

- Pregunta original.
- Consulta contextual utilizada.
- Categoría.
- Respuesta final.
- Resultado de la interacción.
- Estado y confianza de verificación.
- Cantidad y resumen de fuentes.
- Modelo generativo.
- Tiempo total de respuesta.
- Feedback positivo o negativo.
- Errores transitorios.

Los resultados posibles son:

```text
answered
no_evidence
rejected
error
```

El panel **Monitoreo de calidad** muestra:

- Consultas registradas.
- Preguntas sin evidencia.
- Respuestas rechazadas.
- Errores.
- Latencia promedio.
- Feedback positivo y negativo.
- Tasa de feedback.
- Interacciones recientes.
- Posibles vacíos de conocimiento.

Se consideran posibles vacíos:

- Consultas sin evidencia.
- Respuestas rechazadas.
- Errores.
- Respuestas con feedback negativo.

## Persistencia y privacidad

El historial visual del chat se conserva durante la sesión actual de Streamlit.

Las métricas y el feedback se conservan en SQLite después de cerrar la aplicación. Debido a que pueden incluir preguntas y respuestas, la base local no debe subirse a GitHub.

Agregar en `.gitignore`:

```gitignore
storage/faiss_index/
storage/monitoring/
```

Verificar:

```powershell
git check-ignore storage/faiss_index/index.faiss
git check-ignore storage/monitoring/bimbam_quality.db
```

## Fallback y contactos sintéticos

### Sin evidencia

Cuando ningún fragmento supera el umbral:

- No se invoca el modelo generativo.
- No se invoca el verificador.
- Se devuelve un mensaje explícito.
- El estado es `not_applicable`.

### Respuesta rechazada

Cuando el modelo genera contenido no respaldado:

- El verificador marca `rejected`.
- La respuesta se descarta.
- Se muestra un fallback seguro.
- Se registra la interacción para auditoría.

Los contactos alternativos usan direcciones bajo `example.com`:

```text
postventa-bimbam@example.com
```

Son ficticios, no se indexan, no se citan como fuentes y deben sustituirse antes de un uso productivo.

## Consumo de API

Una consulta con evidencia normalmente utiliza:

```text
1 solicitud de embedding
1 solicitud de generación
1 solicitud de verificación
```

Una consulta sin evidencia utiliza únicamente el embedding de la pregunta.

Los fallos transitorios de la verificación se reintentan automáticamente una vez. Si ambos intentos fallan, la interacción se registra como error en SQLite.

## Pruebas manuales realizadas

### Respuesta válida

Consulta:

```text
¿Cuánto tarda un reembolso?
```

Resultado observado:

```text
Estado: verified
Confianza: 1.0
Citas: [1, 2, 3]
Citas inválidas: []
Afirmaciones no respaldadas: []
```

### Respuesta falsa

Afirmación de prueba:

```text
El reembolso tarda exactamente 90 días calendario
y se paga en criptomonedas [Fuente 99].
```

El verificador detectó:

- Plazo no respaldado.
- Medio de pago inventado.
- Fuente inexistente.

### Seguimiento conversacional

Secuencia:

```text
¿Cuánto tarda un reembolso?
¿Y si pagué en cuotas?
```

El asistente utilizó las preguntas recientes para interpretar el seguimiento y volvió a recuperar evidencia documental.


## Mantenimiento documental programable

El pipeline documental ya está integrado con el detector SHA-256.

Comportamiento:

```text
Ejecutar index_documents.py
→ comparar corpus y manifiesto
→ omitir el proceso si no hay cambios
→ reconstruir FAISS si hay cambios
→ guardar corpus_manifest.json
```

Comandos:

```bash
python scripts/index_documents.py
python scripts/index_documents.py --force
```

La interfaz Streamlit compara el corpus actual con el manifiesto y
deshabilita el chat cuando el índice está desactualizado.

El proyecto incluye:

```text
scripts/run_index_maintenance.ps1
scripts/install_windows_index_task.ps1
scripts/run_index_maintenance.sh
scripts/install_linux_cron.sh
```

La tarea programada de Windows fue creada, ejecutada y validada
correctamente durante las pruebas. Después fue eliminada del equipo
local, por lo que la capacidad está implementada pero no permanece
activa en ese computador.

Los scripts de Linux están preparados para activar el mantenimiento
periódico mediante cron cuando el proyecto se despliegue en OCI.


## Banco de evaluación RAG

El proyecto incluye un banco versionado en:

```text
data/evaluation/questions.json
```

Está compuesto por 20 preguntas organizadas en cinco lotes de cuatro
preguntas:

```text
A, B, C, D y E
```

La cobertura incluye:

- Envíos.
- Garantías.
- Reembolsos y devoluciones.
- Métodos de pago.
- Programa de afiliados.
- Preguntas multidocumento.
- Casos fuera de alcance y fallback.

Cada caso registra la pregunta, la categoría, el tipo, la dificultad,
los documentos y páginas esperados, los hechos que deberían aparecer,
las afirmaciones que no deberían generarse y el comportamiento esperado.

### Validación offline

El banco puede validarse sin consumir Gemini:

```bash
python scripts/validate_evaluation_bank.py
python -m pytest tests/test_evaluation_bank.py -v
```

### Evaluador con control de presupuesto

El ejecutor se encuentra en:

```text
scripts/evaluate_rag.py
```

Funciona en modo simulación mientras no se agregue `--execute`.

```bash
python scripts/evaluate_rag.py --batch A --mode retrieval
python scripts/evaluate_rag.py --batch A --mode full
```

Modos disponibles:

| Modo | Reserva conservadora por pregunta | Lote de 4 |
|---|---:|---:|
| `retrieval` | 1 llamada | 4 llamadas |
| `full` | 4 llamadas | 16 llamadas |

El modo completo reserva una llamada para el embedding de consulta, una
para generación, una para verificación y una posible llamada para el
reintento de verificación.

El límite configurado es de 20 llamadas diarias, con un margen de
seguridad predeterminado de dos llamadas. El presupuesto local se registra
en:

```text
storage/evaluation/gemini_budget.json
```

Los resultados de ejecuciones reales se escribirán en:

```text
storage/evaluation/runs/
```

La evaluación semántica de hechos se deja para revisión humana y para el
verificador ya existente. No se utiliza un juez LLM adicional, porque
añadiría otra llamada de Gemini por pregunta.

### Estado de ejecución

La estructura del banco, el validador, las pruebas offline, el planificador
y el contador preventivo ya están implementados. Las evaluaciones reales
no se han ejecutado todavía para conservar la cuota diaria disponible.

## Seguridad y limitaciones

- `.env` no se almacena en Git.
- La clave de Gemini no se muestra.
- El modelo recibe los documentos como contexto, no como instrucciones.
- Las respuestas no verificadas se descartan.
- Los contactos sintéticos están explícitamente marcados.
- La base SQLite local no se sube al repositorio.
- La verificación con otro LLM reduce el riesgo, pero no constituye una garantía matemática.
- Cada verificación agrega latencia y consumo de API.
- El historial completo no se restaura entre sesiones.
- El mantenimiento puede ejecutarse manualmente o programarse; el cron de OCI se activará durante el despliegue.
- El contador de evaluación solo conoce las llamadas reservadas por `evaluate_rag.py`; no puede observar consumos realizados desde Streamlit, la indexación u otros scripts.
- Las evaluaciones reales del banco siguen pendientes de una ventana con cuota suficiente.
- El reranking todavía no forma parte del baseline.

## Próximas etapas

1. Implementar el triaje y la orquestación del agente con LangGraph.
2. Integrar el agente con la interfaz Streamlit sin duplicar la lógica RAG.
3. Ejecutar los lotes del banco cuando la cuota diaria de Gemini esté disponible.
4. Analizar precisión de recuperación, citas, verificación, fallback y latencia.
5. Ajustar el umbral, la confianza mínima y decidir si se necesita reranking.
6. Construir y probar Docker.
7. Desplegar en OCI Compute.
8. Activar el cron de mantenimiento en OCI.

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
  → feedback y monitoreo persistente
  → control de cambios documentales por SHA-256

Banco de evaluación
  → validación offline
  → simulación de presupuesto
  → ejecución controlada por lotes
```

## Autor

Proyecto desarrollado como desafío final de formación en agentes de inteligencia artificial y recuperación aumentada por generación.
