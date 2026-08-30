import os
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Caminho único gravável no ambiente Serverless da Vercel
DB_PATH = "/tmp/pantry.db"

# 2. Configurações de IA
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        pass

MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]

app = FastAPI(title="PantryOS - Inventory & Waste Mitigation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print(f"Erro ao inicializar DB: {e}")

init_db()

# --- SCHEMAS PREDITIVOS ---
class DetectedItem(BaseModel):
    name: str = Field(description="Nome padronizado do ingrediente em português (ex: Queijo Parmesão, Alface Americana)")
    quantity_approx: str = Field(description="Quantidade numérica e unidade (ex: 2 unidades, 300g, 1 maço)")
    category: str = Field(description="Uma entre: protein, produce, dairy, grains, condiments, drinks, bakery, other")
    zone: str = Field(description="Zona de conservação ideal: fridge_shelf, produce_drawer, freezer, pantry")
    shelf_life_days: int = Field(description="Vida útil média em dias na zona indicada")
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

# --- RUNNER COM RETRY ---
def run_ai_structured(contents, schema, prompt):
    global client
    if not client:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada no painel da Vercel.")
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

@app.post("/api/scan")
async def scan_inventory(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        content_type = file.content_type or "image/jpeg"
        if "heic" in content_type.lower() or "heif" in content_type.lower() or content_type == "application/octet-stream":
            content_type = "image/jpeg"

        prompt = """
        Atue como um sistema de auditoria visual de inventário gastronômico.
        Identifique apenas itens comestíveis visíveis com nitidez.
        DIRETRIZES DE EXTRAÇÃO:
        1. Descarte embalagens vazias, fundos indistintos e sombras.
        2. Determine a zona ideal (produce_drawer, fridge_shelf, freezer, pantry).
        3. Apenas itens com confiança HIGH ou MEDIUM devem ser retornados.
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

@app.post("/api/inventory/manual")
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

@app.post("/api/inventory/batch-add")
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

@app.get("/api/inventory")
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

@app.delete("/api/inventory/{item_id}")
def mark_as_consumed(item_id: int):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE inventory SET is_consumed = 1 WHERE id = ?", (item_id,))
        conn.commit()
    return {"status": "ok"}

class RecipeGenRequest(BaseModel):
    mode: str = "quick"

@app.post("/api/generate-recipe")
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
        "preservation": "Elabore uma técnica de pré-preparo, conserva ácida, redução ou técnica de congelamento seguro para impedir a perda dos ingredientes."
    }

    prompt = f"""
    Atue como Chef Executivo e Consultor em Eficiência de Insumos.
    OBJETIVO: {mode_prompts.get(req.mode, mode_prompts['quick'])}
    
    INVENTÁRIO ATIVO:
    - INSUMOS EM RISCO IMINENTE: {', '.join(critical) if critical else 'Nenhum crítico'}
    - INSUMOS EM ESTADO FRESCO: {', '.join(fresh) if fresh else 'Nenhum adicional'}
    - BASE DE DESPENSA: Azeite de oliva, manteiga, alho, cebola, sal, pimenta-do-reino, água.
    
    REGRAS OBRIGATÓRIAS:
    1. Utilize prioritariamente todos os insumos em risco iminente.
    2. Liste os passos com clareza técnica e precisão temporal.
    3. Atribua um Waste Reduction Score (0 a 100) baseado no volume de itens críticos aproveitados.
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

# --- INTERFACE ---
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
      <title>PantryOS — Supply & Culinary Intelligence</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
      <script>
        tailwind.config = {
          darkMode: 'class',
          theme: {
            extend: {
              fontFamily: { sans: ['Inter', 'sans-serif'] },
              colors: {
                zinc: { 950: '#09090b', 900: '#121215', 850: '#18181b', 800: '#27272a', 700: '#3f3f46' },
                accent: { 500: '#10b981', 600: '#059669' }
              }
            }
          }
        }
      </script>
      <style>
        body { background-color: #09090b; color: #e4e4e7; }
        .border-subtle { border-color: rgba(255, 255, 255, 0.08); }
      </style>
    </head>
    <body class="min-h-screen flex justify-center selection:bg-emerald-500/20 selection:text-emerald-400">

      <div class="w-full max-w-4xl px-4 py-8 space-y-8">
        
        <header class="flex items-center justify-between border-b border-subtle pb-6">
          <div class="flex items-center gap-3">
            <div class="h-9 w-9 rounded-lg bg-zinc-800 border border-subtle flex items-center justify-center text-white">
              <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
            </div>
            <div>
              <div class="flex items-center gap-2">
                <h1 class="text-sm font-semibold tracking-tight text-white">PantryOS</h1>
                <span class="text-[10px] font-medium uppercase tracking-widest px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-subtle">v2.4</span>
              </div>
              <p class="text-xs text-zinc-500">Controle de Insumos e Mitigação de Desperdício</p>
            </div>
          </div>

          <div class="flex items-center gap-2.5">
            <button onclick="openManualModal()" class="text-xs font-medium text-zinc-300 hover:text-white bg-zinc-900 border border-subtle px-3.5 py-2 rounded-lg transition hover:bg-zinc-850 flex items-center gap-1.5">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
              Manual
            </button>
            <button onclick="triggerScan()" class="text-xs font-medium text-zinc-950 bg-emerald-400 hover:bg-emerald-300 px-3.5 py-2 rounded-lg transition flex items-center gap-1.5 font-semibold">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
              Escanear Insumos
            </button>
          </div>
        </header>

        <section class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div class="p-4 rounded-xl bg-zinc-900 border border-subtle space-y-1">
            <p class="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Itens Monitorados</p>
            <p id="metricTotal" class="text-2xl font-bold text-white tracking-tight">0</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-900 border border-subtle space-y-1">
            <p class="text-[11px] font-medium uppercase tracking-wider text-amber-500">Expiração Iminente</p>
            <p id="metricWarning" class="text-2xl font-bold text-amber-400 tracking-tight">0</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-900 border border-subtle space-y-1">
            <p class="text-[11px] font-medium uppercase tracking-wider text-rose-500">Risco Crítico (48h)</p>
            <p id="metricCritical" class="text-2xl font-bold text-rose-400 tracking-tight">0</p>
          </div>
          <div class="p-4 rounded-xl bg-zinc-900 border border-subtle space-y-1">
            <p class="text-[11px] font-medium uppercase tracking-wider text-emerald-500">Valor em Risco</p>
            <p id="metricValue" class="text-2xl font-bold text-emerald-400 tracking-tight">R$ 0,00</p>
          </div>
        </section>

        <section class="p-5 rounded-2xl bg-zinc-900 border border-subtle space-y-4">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h2 class="text-sm font-semibold text-white tracking-tight">Planejamento Gastronômico de Mitigação</h2>
              <p class="text-xs text-zinc-400 mt-0.5">Gera preparos focados no aproveitamento imediato dos itens críticos.</p>
            </div>
            <div class="flex items-center gap-2">
              <select id="recipeModeSelect" class="bg-zinc-950 border border-subtle text-xs text-zinc-300 rounded-lg px-3 py-2 focus:outline-none focus:border-zinc-700">
                <option value="quick">Modo Rápido (&le; 20 min)</option>
                <option value="dinner">Menu Estruturado (Refeição)</option>
                <option value="preservation">Técnica de Conservação</option>
              </select>
              <button onclick="requestRecipe()" id="btnRecipeAction" class="text-xs font-semibold text-white bg-zinc-800 hover:bg-zinc-700 border border-subtle px-4 py-2 rounded-lg transition whitespace-nowrap">
                Gerar Ordem
              </button>
            </div>
          </div>
        </section>

        <section class="space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0 text-xs">
              <button onclick="setFilter('all')" class="filter-tab px-3 py-1.5 rounded-lg bg-zinc-800 text-white font-medium border border-subtle" data-filter="all">Todos</button>
              <button onclick="setFilter('critical')" class="filter-tab px-3 py-1.5 rounded-lg text-zinc-400 font-medium hover:text-white" data-filter="critical">Críticos</button>
              <button onclick="setFilter('produce_drawer')" class="filter-tab px-3 py-1.5 rounded-lg text-zinc-400 font-medium hover:text-white" data-filter="produce_drawer">Gaveta Vegetais</button>
              <button onclick="setFilter('fridge_shelf')" class="filter-tab px-3 py-1.5 rounded-lg text-zinc-400 font-medium hover:text-white" data-filter="fridge_shelf">Prateleiras</button>
            </div>
            <input type="text" id="searchInput" oninput="handleSearch(this.value)" placeholder="Buscar insumo..." class="bg-zinc-900 border border-subtle text-xs text-zinc-200 placeholder-zinc-500 rounded-lg px-3 py-2 w-full sm:w-56 focus:outline-none focus:border-zinc-700">
          </div>

          <div id="inventoryContainer" class="space-y-2">
            <div class="text-center py-12 text-zinc-600 text-xs font-medium">Carregando catálogo operacional...</div>
          </div>
        </section>

      </div>

      <input type="file" id="imageScanInput" accept="image/*" class="hidden" onchange="processImageScan(this)">

      <div id="scanReviewModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-zinc-900 w-full max-w-lg rounded-2xl border border-subtle p-6 space-y-4 max-h-[85vh] flex flex-col">
          <div class="flex items-center justify-between pb-3 border-b border-subtle">
            <div>
              <h3 class="text-sm font-semibold text-white">Revisão de Entrada Óptica</h3>
              <p class="text-xs text-zinc-400 mt-0.5">Valide as quantidades e categorias antes do registro.</p>
            </div>
            <button onclick="closeScanModal()" class="text-zinc-500 hover:text-white text-sm">✕</button>
          </div>

          <div id="scannedItemsList" class="overflow-y-auto flex-1 space-y-2 pr-1"></div>

          <div class="pt-2 flex gap-2">
            <button onclick="closeScanModal()" class="w-1/3 bg-zinc-800 hover:bg-zinc-750 text-zinc-300 font-medium py-2.5 rounded-lg text-xs transition">Descartar</button>
            <button onclick="commitScannedItems()" class="w-2/3 bg-emerald-400 hover:bg-emerald-300 text-zinc-950 font-semibold py-2.5 rounded-lg text-xs transition">Confirmar Registro</button>
          </div>
        </div>
      </div>

      <div id="manualAddModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-zinc-900 w-full max-w-md rounded-2xl border border-subtle p-6 space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-subtle">
            <h3 class="text-sm font-semibold text-white">Registro Manual de Insumo</h3>
            <button onclick="closeManualModal()" class="text-zinc-500 hover:text-white text-sm">✕</button>
          </div>

          <form onsubmit="submitManualItem(event)" class="space-y-3 text-xs">
            <div>
              <label class="block text-zinc-400 mb-1">Nome do Insumo</label>
              <input type="text" id="manualName" required class="w-full bg-zinc-950 border border-subtle rounded-lg px-3 py-2 text-white focus:outline-none focus:border-zinc-700">
            </div>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-zinc-400 mb-1">Quantidade</label>
                <input type="text" id="manualQuantity" placeholder="ex: 500g, 2 un" required class="w-full bg-zinc-950 border border-subtle rounded-lg px-3 py-2 text-white focus:outline-none focus:border-zinc-700">
              </div>
              <div>
                <label class="block text-zinc-400 mb-1">Validade Estimada (Dias)</label>
                <input type="number" id="manualDays" value="5" min="1" required class="w-full bg-zinc-950 border border-subtle rounded-lg px-3 py-2 text-white focus:outline-none focus:border-zinc-700">
              </div>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-zinc-400 mb-1">Compartimento</label>
                <select id="manualZone" class="w-full bg-zinc-950 border border-subtle rounded-lg px-3 py-2 text-white focus:outline-none focus:border-zinc-700">
                  <option value="fridge_shelf">Prateleira Refrigerada</option>
                  <option value="produce_drawer">Gaveta de Hortifrúti</option>
                  <option value="freezer">Congelador</option>
                  <option value="pantry">Despensa Seca</option>
                </select>
              </div>
              <div>
                <label class="block text-zinc-400 mb-1">Categoria</label>
                <select id="manualCategory" class="w-full bg-zinc-950 border border-subtle rounded-lg px-3 py-2 text-white focus:outline-none focus:border-zinc-700">
                  <option value="produce">Hortifrúti / Vegetal</option>
                  <option value="protein">Proteína / Carne</option>
                  <option value="dairy">Laticínio</option>
                  <option value="grains">Grãos / Massas</option>
                  <option value="condiments">Molhos e Condimentos</option>
                  <option value="other">Outros</option>
                </select>
              </div>
            </div>
            <div class="pt-2 flex gap-2">
              <button type="button" onclick="closeManualModal()" class="w-1/3 bg-zinc-800 text-zinc-300 py-2.5 rounded-lg">Cancelar</button>
              <button type="submit" class="w-2/3 bg-emerald-400 text-zinc-950 font-semibold py-2.5 rounded-lg">Salvar Insumo</button>
            </div>
          </form>
        </div>
      </div>

      <div id="recipePlanModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-zinc-900 w-full max-w-2xl rounded-2xl border border-subtle p-6 space-y-5 max-h-[90vh] flex flex-col">
          <div class="flex items-center justify-between pb-3 border-b border-subtle">
            <div>
              <div class="flex items-center gap-2">
                <span id="planScoreBadge" class="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">Score 95%</span>
                <span id="planMeta" class="text-xs text-zinc-400">20 min · Fácil</span>
              </div>
              <h3 id="planTitle" class="text-base font-semibold text-white mt-1">Carregando plano...</h3>
            </div>
            <button onclick="closePlanModal()" class="text-zinc-500 hover:text-white text-sm">✕</button>
          </div>

          <div id="planContent" class="overflow-y-auto flex-1 space-y-4 text-xs pr-1"></div>
        </div>
      </div>

      <div id="globalSpinner" class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 hidden flex flex-col items-center justify-center gap-3">
        <div class="w-7 h-7 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
        <p id="spinnerText" class="text-xs font-medium text-zinc-300">Executando computação óptica...</p>
      </div>

      <script>
        let fullInventory = [];
        let stagedItems = [];
        let currentFilter = 'all';

        function triggerScan() {
          document.getElementById('imageScanInput').click();
        }

        async function processImageScan(input) {
          if (!input.files || !input.files[0]) return;
          const file = input.files[0];

          showLoading("Processando visão computacional e extraindo alimentos...");
          const form = new FormData();
          form.append('file', file);

          try {
            const res = await fetch('/api/scan', { method: 'POST', body: form });
            if (!res.ok) throw new Error((await res.json()).detail || "Falha na análise");
            const data = await res.json();
            
            stagedItems = data.items;
            input.value = '';
            hideLoading();
            renderScanReview(stagedItems);
          } catch (e) {
            hideLoading();
            alert("Erro: " + e.message);
          }
        }

        function renderScanReview(items) {
          const list = document.getElementById('scannedItemsList');
          list.innerHTML = '';

          if (!items || items.length === 0) {
            list.innerHTML = '<p class="text-center text-zinc-500 text-xs py-8">Nenhum alimento identificado com nitidez. Tente uma foto com melhor iluminação.</p>';
          } else {
            items.forEach((item, idx) => {
              list.innerHTML += `
                <div class="flex items-center justify-between p-3 rounded-lg bg-zinc-950 border border-subtle">
                  <div class="flex-1 pr-3">
                    <input type="text" value="${item.name}" onchange="stagedItems[${idx}].name = this.value" class="bg-transparent text-xs font-semibold text-white focus:outline-none w-full border-b border-transparent focus:border-zinc-700">
                    <div class="flex items-center gap-2 mt-1">
                      <input type="text" value="${item.quantity_approx}" onchange="stagedItems[${idx}].quantity_approx = this.value" class="bg-transparent text-[11px] text-zinc-400 focus:outline-none w-20">
                      <span class="text-[10px] text-zinc-500">Validade: ${item.shelf_life_days}d</span>
                      <span class="text-[10px] text-zinc-600">(${item.zone})</span>
                    </div>
                  </div>
                  <button onclick="dropStagedItem(${idx})" class="text-zinc-600 hover:text-rose-400 text-xs p-1">✕</button>
                </div>
              `;
            });
          }
          document.getElementById('scanReviewModal').classList.remove('hidden');
        }

        function dropStagedItem(idx) {
          stagedItems.splice(idx, 1);
          renderScanReview(stagedItems);
        }

        function closeScanModal() {
          document.getElementById('scanReviewModal').classList.add('hidden');
          stagedItems = [];
        }

        async function commitScannedItems() {
          if (stagedItems.length === 0) return closeScanModal();

          showLoading("Registrando no banco de dados...");
          try {
            await fetch('/api/inventory/batch-add', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ items: stagedItems })
            });
            closeScanModal();
            hideLoading();
            await loadInventory();
          } catch (e) {
            hideLoading();
            alert("Erro ao persistir: " + e.message);
          }
        }

        function openManualModal() {
          document.getElementById('manualAddModal').classList.remove('hidden');
        }

        function closeManualModal() {
          document.getElementById('manualAddModal').classList.add('hidden');
        }

        async function submitManualItem(e) {
          e.preventDefault();
          const payload = {
            name: document.getElementById('manualName').value,
            quantity: document.getElementById('manualQuantity').value,
            shelf_life_days: parseInt(document.getElementById('manualDays').value),
            zone: document.getElementById('manualZone').value,
            category: document.getElementById('manualCategory').value
          };

          showLoading("Salvando...");
          try {
            await fetch('/api/inventory/manual', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            closeManualModal();
            e.target.reset();
            hideLoading();
            await loadInventory();
          } catch (err) {
            hideLoading();
            alert("Erro: " + err.message);
          }
        }

        async function loadInventory() {
          try {
            const res = await fetch('/api/inventory');
            if (!res.ok) throw new Error("Falha ao obter dados");
            const data = await res.json();
            fullInventory = data.inventory;

            document.getElementById('metricTotal').innerText = data.metrics.total_items;
            document.getElementById('metricWarning').innerText = data.metrics.warning_count;
            document.getElementById('metricCritical').innerText = data.metrics.critical_count;
            document.getElementById('metricValue').innerText = `R$ ${data.metrics.potential_waste_value.toFixed(2)}`;

            renderInventoryList();
          } catch (err) {
            console.error("Erro no loadInventory:", err);
          }
        }

        function renderInventoryList() {
          const container = document.getElementById('inventoryContainer');
          container.innerHTML = '';

          const search = document.getElementById('searchInput').value.toLowerCase();
          
          const filtered = fullInventory.filter(item => {
            const matchesSearch = item.name.toLowerCase().includes(search) || item.category.toLowerCase().includes(search);
            if (!matchesSearch) return false;
            if (currentFilter === 'critical') return item.status === 'critical' || item.status === 'warning';
            if (currentFilter === 'produce_drawer') return item.zone === 'produce_drawer';
            if (currentFilter === 'fridge_shelf') return item.zone === 'fridge_shelf';
            return true;
          });

          if (filtered.length === 0) {
            container.innerHTML = `
              <div class="p-8 text-center rounded-xl border border-dashed border-subtle text-zinc-500 text-xs">
                Nenhum insumo encontrado para a exibição atual.
              </div>
            `;
            return;
          }

          filtered.forEach(item => {
            let statusTag = `<span class="text-[10px] font-medium px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-subtle">${item.days_left}d restantes</span>`;
            let rowBorder = "border-subtle";

            if (item.status === 'critical') {
              statusTag = `<span class="text-[10px] font-semibold px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-900/60">Crítico (${item.days_left}d)</span>`;
              rowBorder = "border-rose-900/40 bg-rose-950/5";
            } else if (item.status === 'warning') {
              statusTag = `<span class="text-[10px] font-medium px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-900/60">Atenção (${item.days_left}d)</span>`;
            }

            container.innerHTML += `
              <div class="flex items-center justify-between p-3.5 rounded-xl bg-zinc-900 border ${rowBorder} hover:border-zinc-700 transition">
                <div class="space-y-0.5">
                  <div class="flex items-center gap-2">
                    <h4 class="text-xs font-semibold text-zinc-100">${item.name}</h4>
                    <span class="text-[10px] text-zinc-500 uppercase tracking-wider">${item.zone.replace('_', ' ')}</span>
                  </div>
                  <p class="text-[11px] text-zinc-400">${item.quantity} · <span class="capitalize text-zinc-500">${item.category}</span></p>
                </div>
                <div class="flex items-center gap-3">
                  ${statusTag}
                  <button onclick="consumeItem(${item.id})" title="Registrar Consumo" class="h-7 w-7 rounded-lg bg-zinc-800 hover:bg-emerald-950 hover:text-emerald-400 hover:border-emerald-800 border border-subtle text-zinc-400 flex items-center justify-center text-xs transition">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                  </button>
                </div>
              </div>
            `;
          });
        }

        function setFilter(f) {
          currentFilter = f;
          document.querySelectorAll('.filter-tab').forEach(b => {
            if (b.dataset.filter === f) {
              b.className = "filter-tab px-3 py-1.5 rounded-lg bg-zinc-800 text-white font-medium border border-subtle";
            } else {
              b.className = "filter-tab px-3 py-1.5 rounded-lg text-zinc-400 font-medium hover:text-white";
            }
          });
          renderInventoryList();
        }

        function handleSearch() {
          renderInventoryList();
        }

        async function consumeItem(id) {
          await fetch(`/api/inventory/${id}`, { method: 'DELETE' });
          await loadInventory();
        }

        async function requestRecipe() {
          const mode = document.getElementById('recipeModeSelect').value;
          showLoading("Otimizando insumos e construindo plano gastronômico...");
          try {
            const res = await fetch('/api/generate-recipe', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ mode })
            });
            if (!res.ok) throw new Error((await res.json()).detail || "Falha na geração");
            const plan = await res.json();

            document.getElementById('planTitle').innerText = plan.recipe_name;
            document.getElementById('planMeta').innerText = `${plan.culinary_style} · ${plan.preparation_time_total} min · ${plan.difficulty_level}`;
            document.getElementById('planScoreBadge').innerText = `Eficiência: ${plan.waste_reduction_score}%`;

            let ingList = plan.ingredients.map(i => `
              <li class="flex items-center justify-between py-1 border-b border-subtle text-zinc-300">
                <span>${i.name} (${i.quantity})</span>
                ${i.is_urgent_item ? '<span class="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-900/60">Prioritário</span>' : ''}
              </li>
            `).join('');

            let stepsList = plan.execution_steps.map((s, idx) => `
              <div class="space-y-0.5">
                <p class="font-semibold text-zinc-200">${idx + 1}. ${s.stage_name} <span class="text-zinc-500 font-normal">(${s.estimated_time_minutes} min)</span></p>
                <p class="text-zinc-400 pl-3 border-l border-zinc-700 leading-relaxed">${s.action}</p>
              </div>
            `).join('');

            document.getElementById('planContent').innerHTML = `
              <div class="space-y-2">
                <h5 class="text-[11px] font-bold uppercase tracking-wider text-zinc-400">Insumos Utilizados</h5>
                <ul class="space-y-0.5">${ingList}</ul>
              </div>

              <div class="space-y-3 pt-2">
                <h5 class="text-[11px] font-bold uppercase tracking-wider text-zinc-400">Instruções de Execução</h5>
                <div class="space-y-2.5">${stepsList}</div>
              </div>

              <div class="p-3 rounded-lg bg-zinc-950 border border-subtle space-y-1">
                <h5 class="text-[11px] font-bold uppercase tracking-wider text-emerald-400">Protocolo de Conservação / Aproveitamento</h5>
                <p class="text-zinc-400 text-xs">${plan.conservation_guide}</p>
              </div>
            `;

            hideLoading();
            document.getElementById('recipePlanModal').classList.remove('hidden');
          } catch (e) {
            hideLoading();
            alert("Erro no planejamento: " + e.message);
          }
        }

        function closePlanModal() {
          document.getElementById('recipePlanModal').classList.add('hidden');
        }

        function showLoading(msg) {
          document.getElementById('spinnerText').innerText = msg;
          document.getElementById('globalSpinner').classList.remove('hidden');
        }

        function hideLoading() {
          document.getElementById('globalSpinner').classList.add('hidden');
        }

        loadInventory();
      </script>
    </body>
    </html>
    """