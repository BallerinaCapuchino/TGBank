# Reattempt: fix string quoting issues and rebuild repo, then zip.

import os, json, textwrap, zipfile, pathlib, uuid, datetime

ROOT = "/mnt/data/mtbank-shop"
FRONT = os.path.join(ROOT, "frontend")
BACK = os.path.join(ROOT, "backend")

def w(path, content):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.strip() + ("\n" if not content.endswith("\n") else ""))

# -----------------------------
# Backend files
# -----------------------------
requirements = """
Flask==3.0.3
Flask-Cors==4.0.1
pydantic==2.9.2
python-dotenv==1.0.1
"""

env_example = """
# Currency shown in UI/back
CURRENCY=MT$
# Price to buy/open a store (in currency units)
STORE_PRICE=1000
# Platform fee in percent (e.g., 5 means 5%)
PLATFORM_FEE_PCT=5
# DB driver: json (default) or sqlite (reserved for future extension)
DB_DRIVER=json
# Telegram bot token to validate initData (optional). If empty, Telegram validation is skipped.
TELEGRAM_BOT_TOKEN=
# Escrow auto-release timeout in hours
ESCROW_TIMEOUT_HOURS=72
# Platform escrow account ID in the bank adapter
PLATFORM_ACCOUNT_ID=platform_escrow
# Flask
FLASK_APP=app.py
FLASK_DEBUG=1
"""

app_py = r"""
import os
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    from routes.auth import auth_bp, load_current_user
    from routes.catalog import catalog_bp
    from routes.stores import stores_bp
    from routes.mystore import mystore_bp
    from routes.orders import orders_bp
    from routes.admin import admin_bp
    from routes.webhooks import webhooks_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(catalog_bp, url_prefix="/api")
    app.register_blueprint(stores_bp, url_prefix="/api")
    app.register_blueprint(mystore_bp, url_prefix="/api")
    app.register_blueprint(orders_bp, url_prefix="/api")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")
    app.register_blueprint(webhooks_bp, url_prefix="/api/webhooks")

    @app.before_request
    def _before():
        load_current_user()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "currency": os.getenv("CURRENCY","MT$")})

    @app.errorhandler(404)
    def nf(e):
        return jsonify({"error":"not_found"}), 404

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
"""

# Adapters
bank_adapter = r"""
import os
from repositories.users_repo import UsersRepo
from repositories.transactions_repo import TransactionsRepo

class BankAdapter:
    # Mock bank adapter over users.json & transactions.json with idempotency
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.users = UsersRepo(data_dir)
        self.tx = TransactionsRepo(data_dir)
        self.platform_account_id = os.getenv("PLATFORM_ACCOUNT_ID", "platform_escrow")

    def get_balance(self, user_id: str) -> float:
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        return round(float(user.get("balance", 0.0)), 2)

    def _apply(self, from_user_id, to_user_id, amount: float, memo: str, idempotency_key: str):
        existing = self.tx.find_by_key(idempotency_key)
        if existing:
            return existing

        if amount <= 0:
            raise ValueError("Amount must be positive")

        if from_user_id:
            from_user = self.users.require(from_user_id)
            if float(from_user["balance"]) < amount:
                raise ValueError("Insufficient funds")
            from_user["balance"] = round(float(from_user["balance"]) - amount, 2)
            self.users.update(from_user_id, from_user)

        to_user = self.users.require(to_user_id)
        to_user["balance"] = round(float(to_user["balance"]) + amount, 2)
        self.users.update(to_user_id, to_user)

        tx = self.tx.create({
            "from_user_id": from_user_id,
            "to_user_id": to_user_id,
            "amount": round(float(amount), 2),
            "memo": memo,
            "idempotency_key": idempotency_key
        })
        return tx

    def transfer(self, from_user_id: str, to_user_id: str, amount: float, memo: str, idempotency_key: str):
        return self._apply(from_user_id, to_user_id, amount, memo, idempotency_key)

    def issue(self, to_user_id: str, amount: float, memo: str, idempotency_key: str):
        return self._apply(None, to_user_id, amount, memo, idempotency_key)

    def platform_account(self) -> str:
        return self.platform_account_id
"""

# Repositories
repo_base = r"""
import os, json, threading, uuid
from typing import List, Dict, Optional, Any

_lock = threading.Lock()

class JsonRepoBase:
    def __init__(self, data_dir: str, file_name: str):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, file_name)
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False)

    def _read_all(self) -> List[Dict[str, Any]]:
        with _lock:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []

    def _write_all(self, data: List[Dict[str, Any]]):
        with _lock:
            tmp = self.file_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.file_path)

    def list(self) -> List[Dict]:
        return self._read_all()

    def get(self, _id: str) -> Optional[Dict]:
        for item in self._read_all():
            if item.get("id") == _id:
                return item
        return None

    def require(self, _id: str) -> Dict:
        found = self.get(_id)
        if not found:
            raise ValueError("Not found")
        return found

    def create(self, item: Dict) -> Dict:
        data = self._read_all()
        if "id" not in item or not item["id"]:
            item["id"] = str(uuid.uuid4())
        data.append(item)
        self._write_all(data)
        return item

    def update(self, _id: str, new_item: Dict) -> Dict:
        data = self._read_all()
        for i, it in enumerate(data):
            if it.get("id") == _id:
                data[i] = new_item
                self._write_all(data)
                return new_item
        raise ValueError("Not found")

    def delete(self, _id: str):
        data = self._read_all()
        data = [it for it in data if it.get("id") != _id]
        self._write_all(data)
"""

users_repo = r"""
from .repo_base import JsonRepoBase

class UsersRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "users.json")

    def find_by_telegram(self, telegram_id: str):
        for u in self.list():
            if str(u.get("telegram_id")) == str(telegram_id):
                return u
        return None
"""

stores_repo = r"""
from .repo_base import JsonRepoBase

class StoresRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "stores.json")

    def list_public(self, include_blocked=False):
        items = self.list()
        if not include_blocked:
            items = [s for s in items if not s.get("is_blocked", False)]
        return items
"""

products_repo = r"""
from .repo_base import JsonRepoBase

class ProductsRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "products.json")

    def list_active_public(self):
        return [p for p in self.list() if p.get("active", True)]
"""

orders_repo = r"""
from .repo_base import JsonRepoBase

class OrdersRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "orders.json")

    def find_by_buyer(self, buyer_id: str):
        return [o for o in self.list() if o.get("buyer_id") == buyer_id]

    def find_by_store(self, store_id: str):
        return [o for o in self.list() if o.get("store_id") == store_id]

    def find_by_key(self, idem_key: str):
        for o in self.list():
            if o.get("idempotency_key") == idem_key:
                return o
        return None
"""

transactions_repo = r"""
from .repo_base import JsonRepoBase

class TransactionsRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "transactions.json")

    def find_by_key(self, idem_key: str):
        for t in self.list():
            if t.get("idempotency_key") == idem_key:
                return t
        return None
"""

notifications_repo = r"""
from .repo_base import JsonRepoBase

class NotificationsRepo(JsonRepoBase):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "notifications.json")
"""

# Models
models_user = r"""
from pydantic import BaseModel
from typing import Optional, Literal

Role = Literal["user","shop_owner","admin"]

class User(BaseModel):
    id: str
    name: str
    role: Role = "user"
    telegram_id: Optional[str] = None
    balance_cached: Optional[float] = None
"""

models_store = r"""
from pydantic import BaseModel
from typing import Optional

class Store(BaseModel):
    id: str
    owner_id: str
    name: str
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    banner_url: Optional[str] = None
    rating: Optional[float] = 4.9
    is_blocked: bool = False
    category: Optional[str] = None
    created_at: str
"""

models_product = r"""
from pydantic import BaseModel, field_validator
from typing import List

class Product(BaseModel):
    id: str
    store_id: str
    title: str
    description: str
    price: float
    stock: int
    images: List[str] = []
    category: str
    active: bool = True

    @field_validator("price")
    def price_positive(cls, v):
        if v <= 0:
            raise ValueError("price must be > 0")
        return round(float(v), 2)

    @field_validator("stock")
    def stock_nonneg(cls, v):
        if v < 0:
            raise ValueError("stock must be >= 0")
        return v
"""

models_order = r"""
from pydantic import BaseModel
from typing import List, Optional, Literal

OrderStatus = Literal["paid","packed","shipped","delivered","released"]

class OrderItem(BaseModel):
    product_id: str
    title: str
    qty: int
    price: float

class Order(BaseModel):
    id: str
    buyer_id: str
    store_id: str
    items: List[OrderItem]
    total: float
    status: OrderStatus
    escrow: bool = True
    created_at: str
    updated_at: str
    idempotency_key: Optional[str] = None
"""

models_tx = r"""
from pydantic import BaseModel
from typing import Optional

class Transaction(BaseModel):
    id: str
    from_user_id: Optional[str] = None
    to_user_id: str
    amount: float
    memo: str
    idempotency_key: str
    created_at: str
"""

# Helpers
util_common = r"""
import hmac, hashlib, urllib.parse, datetime

def now_iso():
    return datetime.datetime.utcnow().isoformat()

def paginate(items, page: int, size: int):
    start = (page-1)*size
    end = start + size
    return items[start:end], len(items)

def parse_int(val, default):
    try:
        return int(val)
    except:
        return default

def parse_float(val, default):
    try:
        return float(val)
    except:
        return default

def verify_telegram_init_data(init_data: str, bot_token: str) -> bool:
    try:
        data = dict([kv.split("=") for kv in init_data.split("&")])
        hash_received = data.pop("hash", None)
        check_arr = []
        for k in sorted(data.keys()):
            check_arr.append(f"{k}={urllib.parse.unquote_plus(data[k])}")
        check_string = "\n".join(check_arr)
        secret_key = hashlib.sha256(bot_token.encode()).digest()
        h = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
        return h == hash_received
    except Exception:
        return False
"""

# Routes
routes_auth = r"""
from flask import Blueprint, request, jsonify, g
import os, json
from adapters.bank_adapter import BankAdapter
from repositories.users_repo import UsersRepo
from utils.common import verify_telegram_init_data

auth_bp = Blueprint("auth", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
bank = BankAdapter(DATA_DIR)
users = UsersRepo(DATA_DIR)

def load_current_user():
    g.user = None
    tg_init = request.headers.get("X-Telegram-Init-Data")
    if tg_init:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN","")
        if bot_token and verify_telegram_init_data(tg_init, bot_token):
            from urllib.parse import parse_qs, unquote_plus
            q = parse_qs(tg_init)
            user_json = unquote_plus(q.get("user", ["{}"])[0])
            user_obj = json.loads(user_json)
            telegram_id = str(user_obj.get("id"))
            u = users.find_by_telegram(telegram_id)
            if not u:
                u = users.create({
                    "name": user_obj.get("first_name","TG User"),
                    "telegram_id": telegram_id,
                    "role": "user",
                    "balance": 0.0
                })
            g.user = u
            return
    auth = request.headers.get("Authorization","")
    if auth.startswith("Bearer "):
        user_id = auth.split(" ",1)[1].strip()
        u = users.get(user_id)
        if u:
            g.user = u

@auth_bp.get("/me")
def me():
    if not g.user:
        return jsonify({"error": "unauthorized"}), 401
    u = dict(g.user)
    try:
        u["balance_cached"] = bank.get_balance(u["id"])
    except Exception:
        pass
    return jsonify(u)
"""

routes_catalog = r"""
from flask import Blueprint, request, jsonify
import os
from repositories.products_repo import ProductsRepo
from repositories.stores_repo import StoresRepo
from utils.common import paginate, parse_int, parse_float

catalog_bp = Blueprint("catalog", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
products = ProductsRepo(DATA_DIR)
stores = StoresRepo(DATA_DIR)

@catalog_bp.get("/catalog")
def catalog():
    search = (request.args.get("search") or "").lower()
    store_id = request.args.get("store_id")
    category = request.args.get("category")
    min_price = parse_float(request.args.get("min"), 0)
    max_price = parse_float(request.args.get("max"), 10**12)
    page = parse_int(request.args.get("page"), 1)
    size = parse_int(request.args.get("size"), 20)

    items = products.list_active_public()
    blocked = {s["id"] for s in stores.list_public(include_blocked=True) if s.get("is_blocked")}
    items = [p for p in items if p.get("store_id") not in blocked]

    if search:
        items = [p for p in items if search in p.get("title","").lower() or search in p.get("description","").lower()]
    if store_id:
        items = [p for p in items if p.get("store_id")==store_id]
    if category:
        items = [p for p in items if p.get("category")==category]
    items = [p for p in items if min_price <= float(p.get("price",0)) <= max_price]

    page_items, total = paginate(items, page, size)
    return jsonify({"items": page_items, "total": total, "page": page, "size": size})
"""

routes_stores = r"""
from flask import Blueprint, request, jsonify, g
import os, uuid, datetime
from adapters.bank_adapter import BankAdapter
from repositories.stores_repo import StoresRepo
from repositories.users_repo import UsersRepo
from utils.common import paginate, parse_int

stores_bp = Blueprint("stores", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
stores = StoresRepo(DATA_DIR)
users = UsersRepo(DATA_DIR)
bank = BankAdapter(DATA_DIR)

@stores_bp.get("/stores")
def list_stores():
    page = parse_int(request.args.get("page"), 1)
    size = parse_int(request.args.get("size"), 20)
    items = stores.list_public(include_blocked=False)
    page_items, total = paginate(items, page, size)
    return jsonify({"items": page_items, "total": total, "page": page, "size": size})

@stores_bp.get("/stores/<store_id>")
def get_store(store_id):
    s = stores.get(store_id)
    if not s or s.get("is_blocked"):
        return jsonify({"error":"not_found_or_blocked"}), 404
    return jsonify(s)

@stores_bp.post("/stores")
def buy_and_create_store():
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    mine = [s for s in stores.list() if s.get("owner_id")==g.user["id"] and not s.get("is_blocked")]
    if len(mine)>0:
        return jsonify({"error":"store_already_exists"}), 400

    data = request.json or {}
    name = data.get("name") or "Мой магазин"
    price = float(os.getenv("STORE_PRICE","1000"))
    idem = data.get("idempotency_key") or str(uuid.uuid4())

    try:
        bank.transfer(g.user["id"], bank.platform_account(), price, f"Buy store {name}", idem)
    except Exception as e:
        return jsonify({"error":"payment_failed","detail":str(e)}), 400

    s = stores.create({
        "owner_id": g.user["id"],
        "name": name,
        "description": data.get("description",""),
        "avatar_url": data.get("avatar_url"),
        "banner_url": data.get("banner_url"),
        "rating": 4.9,
        "is_blocked": False,
        "category": data.get("category","general"),
        "created_at": datetime.datetime.utcnow().isoformat()
    })

    u = users.require(g.user["id"])
    if u.get("role") != "admin":
        u["role"] = "shop_owner"
    users.update(u["id"], u)

    return jsonify(s), 201
"""

routes_mystore = r"""
from flask import Blueprint, request, jsonify, g
import os
from repositories.stores_repo import StoresRepo
from repositories.products_repo import ProductsRepo
from repositories.orders_repo import OrdersRepo

mystore_bp = Blueprint("mystore", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
stores = StoresRepo(DATA_DIR)
products = ProductsRepo(DATA_DIR)
orders = OrdersRepo(DATA_DIR)

def require_owner():
    if not g.user:
        return None, ({"error":"unauthorized"}, 401)
    my = None
    for s in stores.list():
        if s.get("owner_id")==g.user["id"] and not s.get("is_blocked"):
            my = s; break
    if not my:
        return None, ({"error":"no_store"}, 404)
    return my, None

@mystore_bp.get("/mystore")
def my_store():
    s, err = require_owner()
    if err: return err
    return jsonify(s)

@mystore_bp.patch("/mystore")
def update_store():
    s, err = require_owner()
    if err: return err
    data = request.json or {}
    for k in ["name","description","avatar_url","banner_url","category"]:
        if k in data: s[k] = data[k]
    stores.update(s["id"], s)
    return jsonify(s)

@mystore_bp.post("/mystore/products")
def create_product():
    s, err = require_owner()
    if err: return err
    data = request.json or {}
    if float(data.get("price",0))<=0 or int(data.get("stock",0))<0:
        return jsonify({"error":"invalid_price_or_stock"}), 400
    p = products.create({
        "store_id": s["id"],
        "title": data.get("title","Товар"),
        "description": data.get("description",""),
        "price": round(float(data.get("price",1)),2),
        "stock": int(data.get("stock",0)),
        "images": data.get("images", []),
        "category": data.get("category","general"),
        "active": bool(data.get("active", True))
    })
    return jsonify(p), 201

@mystore_bp.patch("/mystore/products/<pid>")
def update_product(pid):
    s, err = require_owner()
    if err: return err
    p = products.get(pid)
    if not p or p.get("store_id")!=s["id"]:
        return jsonify({"error":"not_found"}), 404
    data = request.json or {}
    for k in ["title","description","price","stock","images","category","active"]:
        if k in data: p[k] = data[k]
    if float(p.get("price",0))<=0 or int(p.get("stock",0))<0:
        return jsonify({"error":"invalid_price_or_stock"}), 400
    products.update(pid, p)
    return jsonify(p)

@mystore_bp.delete("/mystore/products/<pid>")
def delete_product(pid):
    s, err = require_owner()
    if err: return err
    p = products.get(pid)
    if not p or p.get("store_id")!=s["id"]:
        return jsonify({"error":"not_found"}), 404
    products.delete(pid)
    return jsonify({"ok":True})

@mystore_bp.get("/mystore/orders")
def store_orders():
    s, err = require_owner()
    if err: return err
    items = [o for o in orders.list() if o.get("store_id")==s["id"]]
    return jsonify({"items":items, "total":len(items), "page":1, "size":len(items)})
"""

routes_orders = r"""
from flask import Blueprint, request, jsonify, g
import os, uuid, datetime
from adapters.bank_adapter import BankAdapter
from repositories.orders_repo import OrdersRepo
from repositories.products_repo import ProductsRepo
from repositories.stores_repo import StoresRepo

orders_bp = Blueprint("orders", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
orders = OrdersRepo(DATA_DIR)
products = ProductsRepo(DATA_DIR)
stores = StoresRepo(DATA_DIR)
bank = BankAdapter(DATA_DIR)

def _compute_total_and_validate(items):
    if not items:
        raise ValueError("Empty items")
    store_id = None
    expanded = []
    total = 0.0
    for it in items:
        p = products.get(it["product_id"])
        if not p or not p.get("active", True):
            raise ValueError("Product not available")
        st = stores.get(p["store_id"])
        if st.get("is_blocked"):
            raise ValueError("Store is blocked")
        if store_id is None:
            store_id = p["store_id"]
        if p["store_id"] != store_id:
            raise ValueError("All items must be from the same store")
        qty = int(it.get("qty",1))
        if qty < 1:
            raise ValueError("qty >= 1 required")
        if int(p["stock"]) < qty:
            raise ValueError(f"Insufficient stock for {p['title']}")
        total += float(p["price"]) * qty
        expanded.append({"product_id": p["id"], "title": p["title"], "qty": qty, "price": float(p["price"])})
    return store_id, expanded, round(total, 2)

@orders_bp.post("/orders/checkout")
def checkout():
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    data = request.json or {}
    items = data.get("items", [])
    idem = data.get("idempotency_key") or str(uuid.uuid4())
    exist = orders.find_by_key(idem)
    if exist:
        return jsonify(exist)

    try:
        store_id, expanded, total = _compute_total_and_validate(items)
    except Exception as e:
        return jsonify({"error":"validation_failed","detail":str(e)}), 400

    try:
        bank.transfer(g.user["id"], bank.platform_account(), total, f"Order escrow for store {store_id}", idem)
    except Exception as e:
        return jsonify({"error":"payment_failed","detail":str(e)}), 400

    for it in expanded:
        p = products.require(it["product_id"])
        if int(p["stock"]) < int(it["qty"]):
            return jsonify({"error":"race_stock"}), 409
        p["stock"] = int(p["stock"]) - int(it["qty"])
        products.update(p["id"], p)

    o = orders.create({
        "buyer_id": g.user["id"],
        "store_id": store_id,
        "items": expanded,
        "total": total,
        "status": "paid",
        "escrow": True,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "idempotency_key": idem
    })
    return jsonify(o), 201

@orders_bp.get("/orders/my")
def my_orders():
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    items = [o for o in orders.list() if o.get("buyer_id")==g.user["id"]]
    return jsonify({"items":items, "total":len(items), "page":1, "size":len(items)})

@orders_bp.post("/orders/<oid>/ship")
def ship(oid):
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    o = orders.get(oid)
    if not o: return jsonify({"error":"not_found"}), 404
    s = stores.require(o["store_id"])
    if s["owner_id"] != g.user["id"]:
        return jsonify({"error":"forbidden"}), 403
    if o["status"] not in ["paid","packed"]:
        return jsonify({"error":"invalid_status"}), 400
    o["status"] = "shipped"
    o["updated_at"] = datetime.datetime.utcnow().isoformat()
    orders.update(o["id"], o)
    return jsonify(o)

@orders_bp.post("/orders/<oid>/confirm-delivery")
def confirm_delivery(oid):
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    o = orders.get(oid)
    if not o: return jsonify({"error":"not_found"}), 404
    if o["buyer_id"] != g.user["id"]:
        return jsonify({"error":"forbidden"}), 403
    if o["status"] not in ["shipped","delivered"]:
        return jsonify({"error":"invalid_status"}), 400

    fee_pct = float(os.getenv("PLATFORM_FEE_PCT","5"))
    seller = stores.require(o["store_id"])["owner_id"]
    amount = round(float(o["total"]) * (1.0 - fee_pct/100.0), 2)
    idem = str(uuid.uuid4())
    bank.transfer(bank.platform_account(), seller, amount, f"Escrow release for order {o['id']}", idem)

    o["status"] = "released"
    o["escrow"] = False
    o["updated_at"] = datetime.datetime.utcnow().isoformat()
    orders.update(o["id"], o)
    return jsonify(o)

@orders_bp.post("/orders/<oid>/open-dispute")
def open_dispute(oid):
    if not g.user:
        return jsonify({"error":"unauthorized"}), 401
    o = orders.get(oid)
    if not o: return jsonify({"error":"not_found"}), 404
    from repositories.notifications_repo import NotificationsRepo
    notes = NotificationsRepo(DATA_DIR)
    notes.create({"user_id": o["store_id"], "type":"dispute", "payload":{"order_id": oid, "buyer_id": g.user["id"]}, "read": False})
    return jsonify({"ok": True, "message":"Поддержка получила ваш запрос"})
"""

routes_admin = r"""
from flask import Blueprint, request, jsonify, g
import os
from repositories.stores_repo import StoresRepo
from repositories.products_repo import ProductsRepo
from repositories.orders_repo import OrdersRepo
from repositories.users_repo import UsersRepo

admin_bp = Blueprint("admin", __name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
stores = StoresRepo(DATA_DIR)
products = ProductsRepo(DATA_DIR)
orders = OrdersRepo(DATA_DIR)
users = UsersRepo(DATA_DIR)

def require_admin():
    return bool(g.user and g.user.get("role") == "admin")

@admin_bp.get("/stores")
def admin_stores():
    if not require_admin(): return ({"error":"forbidden"}, 403)
    status = request.args.get("status")
    items = stores.list()
    if status == "blocked":
        items = [s for s in items if s.get("is_blocked")]
    elif status == "active":
        items = [s for s in items if not s.get("is_blocked")]
    return jsonify({"items":items, "total": len(items)})

@admin_bp.patch("/stores/<sid>/block")
def block_store(sid):
    if not require_admin(): return ({"error":"forbidden"}, 403)
    data = request.json or {}
    s = stores.get(sid)
    if not s: return ({"error":"not_found"},404)
    s["is_blocked"] = bool(data.get("is_blocked", True))
    stores.update(sid, s)
    return jsonify(s)

@admin_bp.get("/reports/summary")
def summary():
    if not require_admin(): return ({"error":"forbidden"}, 403)
    return jsonify({
        "orders": len(orders.list()),
        "products": len(products.list()),
        "stores": len(stores.list()),
        "users": len(users.list())
    })
"""

routes_webhooks = r"""
from flask import Blueprint, jsonify
webhooks_bp = Blueprint("webhooks", __name__)

@webhooks_bp.post("/bank")
def bank_sync():
    return jsonify({"ok": True})
"""

# Seeds
seed_users = [
    {"id":"1","name":"Max (ADMIN)","role":"admin","balance":0.0},
    {"id":"2","name":"Алиса (Buyer)","role":"user","balance":5000.0},
    {"id":"3","name":"Борис (Seller)","role":"shop_owner","balance":200.0},
    {"id":"platform_escrow","name":"Platform Escrow","role":"user","balance":0.0},
]

seed_stores = [
    {"id":"store-1","owner_id":"3","name":"Boris Shop","description":"Гаджеты и аксессуары","avatar_url":"","banner_url":"","rating":4.9,"is_blocked":False,"category":"electronics","created_at":datetime.datetime.utcnow().isoformat()},
    {"id":"store-2","owner_id":"3","name":"Boris Outdoors","description":"Туризм и природа","avatar_url":"","banner_url":"","rating":4.8,"is_blocked":False,"category":"outdoors","created_at":datetime.datetime.utcnow().isoformat()},
    {"id":"store-3","owner_id":"1","name":"Admin Goods","description":"Официальный магазин","avatar_url":"","banner_url":"","rating":5.0,"is_blocked":False,"category":"general","created_at":datetime.datetime.utcnow().isoformat()},
]

def mk_products_for_store(sid, prefix):
    items = []
    for i in range(1,9):
        items.append({
            "id": f"{sid}-p{i}",
            "store_id": sid,
            "title": f"{prefix} Товар {i}",
            "description": "Качественный товар с красивым описанием.",
            "price": round(50 + i*10.25, 2),
            "stock": 5 + i,
            "images": ["https://picsum.photos/seed/"+str(uuid.uuid4())+"/600/400"],
            "category": "general",
            "active": True
        })
    return items

seed_products = mk_products_for_store("store-1","Boris") + mk_products_for_store("store-2","Outdoors") + mk_products_for_store("store-3","Admin")


