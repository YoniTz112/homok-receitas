import os
import time
import sqlite3
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from google import genai
from google.genai import types

# ============================================================
# PANTRYOS - BACKEND COMPLETO
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pantryos")

DB_PATH = os.environ.get("DB_PATH", "/tmp/pantry.db")
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB

# Gemini 3.8 Flash é o modelo principal. Flash-Lite fica como fallback.
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

client = None

app = FastAPI(title="PantryOS Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VERCEL ROUTE NORMALIZATION
# ============================================================

@app.middleware("http")
async def normalize_vercel_routes(request: Request, call_next):
    path = request.scope.get("path", "")
    prefixes = ("/api/index.py", "/api/index")

    for prefix in prefixes:
        if path == prefix:
            request.scope["path"] = "/"
            break
        if path.startswith(prefix + "/"):
            request.scope["path"] = path[len(prefix):]
            break

    return await call_next(request)


# ============================================================
# DATABASE
# ============================================================

def init_db():
    try:
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(DB_PATH, timeout=20) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    zone TEXT NOT NULL DEFAULT 'fridge_shelf',
                    added_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    estimated_value REAL DEFAULT 10.0,
                    is_consumed INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()
    except Exception:
        logger.exception("Erro ao inicializar banco de dados")
        raise


init_db()


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class DetectedItem(BaseModel):
    name: str = Field(description="Nome padronizado do alimento no idioma solicitado")
    quantity_approx: str = Field(description="Quantidade aproximada, por exemplo 2 un ou 500 g")
    category: str = Field(description="Uma destas categorias: produce, protein, dairy, grains, condiments, other")
    zone: str = Field(description="Uma destas zonas: fridge_shelf, produce_drawer, freezer, pantry")
    shelf_life_days: int = Field(description="Estimativa de dias de conservação")
    confidence: str = Field(description="HIGH ou MEDIUM")


class VisionScanResponse(BaseModel):
    detected_items: List[DetectedItem]


class RecipeIngredient(BaseModel):
    name: str
    quantity: str
    is_urgent_item: bool


class RecipeStep(BaseModel):
    stage_name: str
    action: str
    estimated_time_minutes: int


class CulinaryPlanResponse(BaseModel):
    recipe_name: str
    culinary_style: str
    preparation_time_total: int
    difficulty_level: str
    waste_reduction_score: int
    ingredients: List[RecipeIngredient]
    execution_steps: List[RecipeStep]
    conservation_guide: str


class ManualItemInput(BaseModel):
    name: str
    quantity: str
    category: str
    zone: str
    shelf_life_days: int = Field(ge=0, le=3650)
    estimated_value: Optional[float] = 12.0


class BatchAddRequest(BaseModel):
    items: List[DetectedItem]


class RecipeGenRequest(BaseModel):
    mode: str = "quick"
    lang: Optional[str] = "pt"


# ============================================================
# GEMINI CLIENT + ERROR HANDLING
# ============================================================

def get_gemini_client():
    global client

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY não está configurada nas Environment Variables da Vercel."
        )

    if client is None:
        try:
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.exception("Não foi possível criar o cliente Gemini")
            raise HTTPException(status_code=503, detail=f"Falha ao iniciar Gemini: {str(exc)[:400]}")

    return client


def classify_gemini_error(error: Exception) -> HTTPException:
    message = str(error)
    lower = message.lower()

    logger.error("Gemini error: %s", message)

    if "429" in message or "resource_exhausted" in lower or "quota" in lower:
        return HTTPException(
            status_code=429,
            detail="Limite ou cota da Gemini API atingido. Confira o uso e o billing do projeto no Google AI Studio."
        )

    if "401" in message or "403" in message or "api key" in lower or "permission_denied" in lower:
        return HTTPException(
            status_code=502,
            detail="A Gemini API recusou a chave ou a permissão. Confira GEMINI_API_KEY na Vercel."
        )

    if "404" in message or "not_found" in lower or "not found" in lower:
        return HTTPException(
            status_code=502,
            detail="Modelo Gemini não encontrado ou indisponível para esta API. Confira o nome do modelo e a versão do SDK."
        )

    if "503" in message or "unavailable" in lower or "overloaded" in lower:
        return HTTPException(
            status_code=503,
            detail="Gemini está temporariamente indisponível ou sobrecarregado. Tente novamente em alguns segundos."
        )

    if "timeout" in lower or "timed out" in lower:
        return HTTPException(
            status_code=504,
            detail="A Gemini API demorou demais para responder. Tente uma imagem menor."
        )

    return HTTPException(status_code=502, detail=f"Erro na Gemini API: {message[:700]}")


def run_ai_structured(contents, schema, prompt):
    """Executa Gemini com saída JSON estruturada e fallback de modelo."""
    gemini = get_gemini_client()
    last_error = None

    for model_name in MODELS_TO_TRY:
        try:
            response = gemini.models.generate_content(
                model=model_name,
                contents=contents + [prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                    candidate_count=1,
                ),
            )

            result_text = getattr(response, "text", None)
            if not result_text or not result_text.strip():
                raise ValueError(f"O modelo {model_name} retornou uma resposta vazia.")

            return result_text

        except Exception as exc:
            last_error = exc
            logger.exception("Falha usando modelo %s", model_name)

            # Só tenta novamente em erros temporários. Não fica repetindo
            # chamadas em erros de chave, cota ou schema.
            msg = str(exc).lower()
            temporary = any(token in msg for token in ["503", "unavailable", "overloaded", "429", "resource_exhausted"])
            if temporary and model_name == MODELS_TO_TRY[0]:
                time.sleep(1)
                continue

            # Erros de modelo podem ser resolvidos pelo modelo alternativo.
            model_problem = any(token in msg for token in ["404", "not found", "not_found", "unsupported"])
            if model_problem and model_name == MODELS_TO_TRY[0]:
                continue

            # Para erros permanentes, não gasta outra chamada.
            if not temporary and not model_problem:
                break

    if last_error:
        raise classify_gemini_error(last_error)

    raise HTTPException(status_code=502, detail="Não foi possível obter resposta da Gemini API.")


# ============================================================
# IMAGE MIME DETECTION
# ============================================================

def detect_image_mime(image_bytes: bytes, supplied_type: Optional[str]) -> str:
    """Detecta o formato real pelos bytes; não finge que HEIC é JPEG."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="A imagem enviada está vazia.")

    # JPEG
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    # PNG
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    # WEBP
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    # HEIC / HEIF (ISO Base Media File Format)
    if len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        brand = image_bytes[8:12].decode("ascii", errors="ignore").lower()
        if brand in {"heic", "heix", "hevc", "hevx", "mif1", "msf1"}:
            return "image/heic" if brand.startswith("hei") else "image/heif"

    # Se o navegador mandou um tipo conhecido, aceita apenas formatos seguros.
    supplied = (supplied_type or "").lower().split(";")[0].strip()
    aliases = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}
    supplied = aliases.get(supplied, supplied)

    allowed = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
    if supplied in allowed:
        return supplied

    raise HTTPException(
        status_code=415,
        detail="Formato não suportado. Envie uma imagem JPG, PNG, WEBP ou HEIC."
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
@app.get("/api")
def health_check():
    return {"status": "ok", "service": "PantryOS Engine", "version": "2.0.0"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "healthy", "gemini_key_configured": bool(os.environ.get("GEMINI_API_KEY"))}


# ============================================================
# SCAN INVENTORY
# ============================================================

@app.post("/scan")
@app.post("/api/scan")
async def scan_inventory(file: UploadFile = File(...), lang: str = "pt"):
    try:
        image_bytes = await file.read()

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Imagem muito grande. O limite é 12 MB.")

        content_type = detect_image_mime(image_bytes, file.content_type)

        if lang == "uy":
            lang_instruction = (
                "Retorne os nomes dos alimentos estritamente em ESPANHOL DO URUGUAI / Rioplatense "
                "(ex: morrón, frutillas, chauchas, manteca, zapallito, boniato, palta, dulce de leche)."
            )
        else:
            lang_instruction = "Retorne os nomes dos alimentos estritamente em PORTUGUÊS DO BRASIL."

        prompt = f"""
Você é o sistema de visão computacional do PantryOS.
Analise a imagem e identifique apenas alimentos comestíveis visíveis com clareza.
Não invente alimentos escondidos, embalagens ilegíveis ou objetos que não sejam comida.
Para cada alimento, estime quantidade, categoria, local ideal de armazenamento e validade aproximada.
Use shelf_life_days como número inteiro entre 0 e 365.
Use category somente: produce, protein, dairy, grains, condiments, other.
Use zone somente: fridge_shelf, produce_drawer, freezer, pantry.
Use confidence somente: HIGH ou MEDIUM.
Retorne apenas itens com confiança HIGH ou MEDIUM.
Se não houver alimento identificável, retorne detected_items vazio.
{lang_instruction}
"""

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=content_type)
        ]

        raw_json = await run_in_threadpool(
            run_ai_structured,
            contents,
            VisionScanResponse,
            prompt,
        )

        parsed = VisionScanResponse.model_validate_json(raw_json)
        return {"items": parsed.detected_items}

    except HTTPException:
        raise
    except ValidationError as exc:
        logger.exception("JSON retornado pelo Gemini não passou na validação")
        raise HTTPException(status_code=502, detail=f"Resposta inválida da IA: {str(exc)[:500]}")
    except Exception as exc:
        logger.exception("Erro inesperado no endpoint /scan")
        raise HTTPException(status_code=500, detail=f"Erro interno no scan: {str(exc)[:700]}")
    finally:
        await file.close()


# ============================================================
# ADD ITEM MANUALLY
# ============================================================

@app.post("/inventory/manual")
@app.post("/api/inventory/manual")
def add_manual_item(item: ManualItemInput):
    init_db()
    now = datetime.now()
    expires_at = now + timedelta(days=item.shelf_life_days)

    with sqlite3.connect(DB_PATH, timeout=20) as conn:
        conn.execute(
            """
            INSERT INTO inventory
            (name, quantity, category, zone, added_at, expires_at, estimated_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.name.strip().title(), item.quantity, item.category,
                item.zone, now.isoformat(), expires_at.isoformat(),
                item.estimated_value if item.estimated_value is not None else 12.0,
            ),
        )
        conn.commit()

    return {"status": "ok"}


# ============================================================
# ADD SCANNED ITEMS IN BATCH
# ============================================================

@app.post("/inventory/batch-add")
@app.post("/api/inventory/batch-add")
def batch_add_items(req: BatchAddRequest):
    init_db()
    now = datetime.now()

    with sqlite3.connect(DB_PATH, timeout=20) as conn:
        for item in req.items:
            expires_at = now + timedelta(days=max(0, item.shelf_life_days))
            conn.execute(
                """
                INSERT INTO inventory
                (name, quantity, category, zone, added_at, expires_at, estimated_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.name.strip().title(), item.quantity_approx,
                    item.category, item.zone, now.isoformat(),
                    expires_at.isoformat(), 15.0,
                ),
            )
        conn.commit()

    return {"status": "ok", "added": len(req.items)}


# ============================================================
# GET INVENTORY
# ============================================================

@app.get("/inventory")
@app.get("/api/inventory")
def get_inventory():
    init_db()
    now = datetime.now()

    with sqlite3.connect(DB_PATH, timeout=20) as conn:
        rows = conn.execute(
            """
            SELECT id, name, quantity, category, zone, added_at,
                   expires_at, estimated_value
            FROM inventory
            WHERE is_consumed = 0
            """
        ).fetchall()

    items = []
    for row in rows:
        try:
            exp_date = datetime.fromisoformat(row[6])
        except (TypeError, ValueError):
            exp_date = now

        hours_left = (exp_date - now).total_seconds() / 3600
        days_left = max(0, int(hours_left // 24))

        if hours_left <= 48:
            status = "critical"
        elif hours_left <= 120:
            status = "warning"
        else:
            status = "fresh"

        value = float(row[7] or 0.0)
        items.append({
            "id": row[0],
            "name": row[1],
            "quantity": row[2],
            "category": row[3],
            "zone": row[4],
            "days_left": days_left,
            "status": status,
            "value": value,
        })

    items.sort(key=lambda item: item["days_left"])

    return {
        "inventory": items,
        "metrics": {
            "total_items": len(items),
            "critical_count": sum(1 for item in items if item["status"] == "critical"),
            "warning_count": sum(1 for item in items if item["status"] == "warning"),
            "fresh_count": sum(1 for item in items if item["status"] == "fresh"),
            "potential_waste_value": round(
                sum(item["value"] for item in items if item["status"] in ["critical", "warning"]), 2
            ),
        },
    }


# ============================================================
# MARK ITEM AS CONSUMED
# ============================================================

@app.delete("/inventory/{item_id}")
@app.delete("/api/inventory/{item_id}")
def mark_as_consumed(item_id: int):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=20) as conn:
        cursor = conn.execute(
            "UPDATE inventory SET is_consumed = 1 WHERE id = ?",
            (item_id,),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Item não encontrado.")

    return {"status": "ok"}


# ============================================================
# GENERATE RECIPE
# ============================================================

@app.post("/generate-recipe")
@app.post("/api/generate-recipe")
async def generate_recipe(req: RecipeGenRequest):
    data = get_inventory()
    inventory = data["inventory"]

    if not inventory:
        raise HTTPException(status_code=400, detail="Inventário vazio.")

    critical = [
        f"{item['name']} ({item['quantity']})"
        for item in inventory if item["status"] in ["critical", "warning"]
    ]
    fresh = [
        f"{item['name']} ({item['quantity']})"
        for item in inventory if item["status"] == "fresh"
    ]

    is_uy = req.lang == "uy"

    if req.mode == "dinner":
        mode_prompt = (
            "Elaborá un plato principal completo y bien resuelto para la cena o el almuerzo."
            if is_uy else "Elabore um prato principal completo e bem estruturado para almoço ou jantar."
        )
    elif req.mode == "preservation":
        mode_prompt = (
            "Elaborá una técnica segura de pre-elaboración, conservación o congelado."
            if is_uy else "Elabore uma técnica segura de pré-preparo, conservação ou congelamento."
        )
    else:
        mode_prompt = (
            "Priorizá una receta rápida de máximo 20 minutos."
            if is_uy else "Priorize uma receita rápida de no máximo 20 minutos."
        )

    lang_rules = (
        "Escreva todo o conteúdo em ESPANHOL DO URUGUAI (Rioplatense; use termos como morrón, manteca, chauchas e boniato)."
        if is_uy else "Escreva todo o conteúdo em PORTUGUÊS DO BRASIL."
    )

    prompt = f"""
Atue como Chef Executivo do PantryOS.
OBJETIVO: {mode_prompt}
IDIOMA: {lang_rules}

INVENTÁRIO ATIVO:
- INSUMOS URGENTES: {', '.join(critical) if critical else 'Nenhum'}
- INSUMOS FRESCOS: {', '.join(fresh) if fresh else 'Nenhum'}
- BÁSICOS QUE PODEM SER CONSIDERADOS: azeite, manteiga, alho, cebola, sal, pimenta e água.

Priorize usar os ingredientes urgentes sem inventar ingredientes caros ou incomuns.
Não sugira técnicas de conserva inseguras. waste_reduction_score deve ficar entre 0 e 100.
"""

    raw_json = await run_in_threadpool(
        run_ai_structured,
        [],
        CulinaryPlanResponse,
        prompt,
    )

    try:
        return CulinaryPlanResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        logger.exception("Resposta de receita inválida")
        raise HTTPException(status_code=502, detail=f"Resposta inválida da IA: {str(exc)[:500]}")


