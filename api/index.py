import os
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

DB_PATH = "/tmp/pantry.db"

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        pass

MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]

app = FastAPI(title="PantryOS Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para normalizar prefixos de rotas enviados pela Vercel
@app.middleware("http")
async def normalize_vercel_routes(request: Request, call_next):
    path = request.scope.get("path", "")
    for prefix in ["/api/index.py", "/api/index", "/api"]:
        if path.startswith(prefix) and len(path) > len(prefix):
            request.scope["path"] = path[len(prefix):]
            break
    response = await call_next(request)
    return response

def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
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
            """)
            conn.commit()
    except Exception as e:
        print(f"DB Init error: {e}")

init_db()

# --- SCHEMAS ---
class DetectedItem(BaseModel):
    name: str = Field(description="Nome padronizado do ingrediente em português")
    quantity_approx: str = Field(description="Quantidade numérica e unidade")
    category: str = Field(description="Uma entre: protein, produce, dairy, grains, condiments, drinks, bakery, other")
    zone: str = Field(description="Zona ideal: fridge_shelf, produce_drawer, freezer, pantry")
    shelf_life_days: int = Field(description="Vida útil média em dias")
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

def run_ai_structured(contents, schema, prompt):
    global client
    if not client:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
        client = genai.Client(api_key=api_key)
        
    last_err = None
    for model_name in MODELS_TO_TRY:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents + [prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
                return response.text
            except Exception as err:
                last_err = err
                if "503" in str(err) or "UNAVAILABLE" in str(err):
                    time.sleep((attempt + 1) * 2)
                else:
                    break
    raise last_err

# --- ENDPOINTS ---

@app.post("/scan")
async def scan_inventory(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        content_type = file.content_type or "image/jpeg"
        if "heic" in content_type.lower() or "heif" in content_type.lower() or content_type == "application/octet-stream":
            content_type = "image/jpeg"

        prompt = """
        Identifique apenas alimentos visíveis em primeiro plano com clareza.
        Determine a zona ideal (produce_drawer, fridge_shelf, freezer, pantry).
        Retorne apenas itens com confiança HIGH ou MEDIUM.
        """

        raw_json = run_ai_structured(
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=content_type)],
            schema=VisionScanResponse,
            prompt=prompt
        )
        data = VisionScanResponse.model_validate_json(raw_json)
        return {"items": data.detected_items}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ManualItemInput(BaseModel):
    name: str
    quantity: str
    category: str
    zone: str
    shelf_life_days: int
    estimated_value: Optional[float] = 12.0

@app.post("/inventory/manual")
def add_manual_item(item: ManualItemInput):
    init_db()
    now = datetime.now()
    expires_at = now + timedelta(days=item.shelf_life_days)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inventory (name, quantity, category, zone, added_at, expires_at, estimated_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (item.name.strip().title(), item.quantity, item.category, item.zone, now.isoformat(), expires_at.isoformat(), item.estimated_value))
        conn.commit()
    return {"status": "ok"}

class BatchAddRequest(BaseModel):
    items: List[DetectedItem]

@app.post("/inventory/batch-add")
def batch_add_items(req: BatchAddRequest):
    init_db()
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for item in req.items:
            expires_at = now + timedelta(days=item.shelf_life_days)
            cursor.execute("""
                INSERT INTO inventory (name, quantity, category, zone, added_at, expires_at, estimated_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (item.name.strip().title(), item.quantity_approx, item.category, item.zone, now.isoformat(), expires_at.isoformat(), 15.0))
        conn.commit()
    return {"status": "ok"}

@app.get("/inventory")
def get_inventory():
    init_db()
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, quantity, category, zone, added_at, expires_at, estimated_value FROM inventory WHERE is_consumed = 0")
        rows = cursor.fetchall()

    items = []
    for r in rows:
        exp_date = datetime.fromisoformat(r[6])
        hours_left = (exp_date - now).total_seconds() / 3600
        days_left = max(0, int(hours_left // 24))

        if hours_left <= 48:
            status = "critical"
        elif hours_left <= 120:
            status = "warning"
        else:
            status = "fresh"

        val = r[7] if r[7] else 0.0

        items.append({
            "id": r[0],
            "name": r[1],
            "quantity": r[2],
            "category": r[3],
            "zone": r[4],
            "days_left": days_left,
            "status": status,
            "value": val
        })

    items.sort(key=lambda x: x["days_left"])
    return {
        "inventory": items,
        "metrics": {
            "total_items": len(items),
            "critical_count": sum(1 for i in items if i["status"] == "critical"),
            "warning_count": sum(1 for i in items if i["status"] == "warning"),
            "fresh_count": sum(1 for i in items if i["status"] == "fresh"),
            "potential_waste_value": sum(i["value"] for i in items if i["status"] in ["critical", "warning"])
        }
    }

@app.delete("/inventory/{item_id}")
def mark_as_consumed(item_id: int):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE inventory SET is_consumed = 1 WHERE id = ?", (item_id,))
        conn.commit()
    return {"status": "ok"}

class RecipeGenRequest(BaseModel):
    mode: str = "quick"

@app.post("/generate-recipe")
def generate_recipe(req: RecipeGenRequest):
    data = get_inventory()
    inv = data["inventory"]
    if not inv:
        raise HTTPException(status_code=400, detail="Inventário vazio.")

    critical = [f"{i['name']} ({i['quantity']})" for i in inv if i["status"] in ["critical", "warning"]]
    fresh = [f"{i['name']} ({i['quantity']})" for i in inv if i["status"] == "fresh"]

    mode_prompts = {
        "quick": "Priorize uma preparação ágil de no máximo 20 minutos.",
        "dinner": "Elabore um prato principal estruturado e refinado.",
        "preservation": "Elabore uma técnica de pré-preparo, conserva ácida ou congelamento seguro."
    }

    prompt = f"""
    Atue como Chef Executivo.
    OBJETIVO: {mode_prompts.get(req.mode, mode_prompts['quick'])}
    
    INVENTÁRIO ATIVO:
    - INSUMOS CRÍTICOS: {', '.join(critical) if critical else 'Nenhum crítico'}
    - INSUMOS FRESCOS: {', '.join(fresh) if fresh else 'Nenhum adicional'}
    - DESPENSA BÁSICA: Azeite de oliva, manteiga, alho, cebola, sal, pimenta-do-reino, água.
    """

    try:
        raw_json = run_ai_structured(
            contents=[],
            schema=CulinaryPlanResponse,
            prompt=prompt
        )
        return CulinaryPlanResponse.model_validate_json(raw_json)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))