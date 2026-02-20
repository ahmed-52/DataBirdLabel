import io
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from rasterio.windows import Window

TILE_SIZE = 1280
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_ROOT = BASE_DIR / "static"
DATA_ROOT = BASE_DIR / "data"
UPLOAD_ROOT = DATA_ROOT / "uploads"
DB_PATH = DATA_ROOT / "databirdlabel.db"

STATIC_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="DataBLaber API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


class ClassCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class AnnotationCreate(BaseModel):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = None
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "project"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS label_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orthomosaics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                tile_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                orthomosaic_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                row_idx INTEGER NOT NULL,
                col_idx INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (orthomosaic_id) REFERENCES orthomosaics(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                x_center REAL NOT NULL,
                y_center REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (tile_id) REFERENCES tiles(id) ON DELETE CASCADE,
                FOREIGN KEY (class_id) REFERENCES label_classes(id)
            );
            """
        )


@app.on_event("startup")
def startup() -> None:
    init_db()


def ensure_project(project_id: int) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def tile_to_uint8(tile: np.ndarray) -> np.ndarray:
    if tile.size == 0:
        return tile.astype(np.uint8)

    if tile.dtype == np.uint8:
        return tile

    # Normalize each tile for image-friendly jpg output.
    tile_f = tile.astype(np.float32)
    t_min = float(tile_f.min())
    t_max = float(tile_f.max())
    if t_max <= t_min:
        return np.zeros_like(tile_f, dtype=np.uint8)
    scaled = (tile_f - t_min) / (t_max - t_min)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def raster_to_rgb(tile_chw: np.ndarray) -> np.ndarray:
    bands, height, width = tile_chw.shape
    if bands >= 3:
        rgb = np.moveaxis(tile_chw[:3], 0, -1)
    elif bands == 1:
        mono = tile_chw[0]
        rgb = np.stack([mono, mono, mono], axis=-1)
    else:
        rgb = np.zeros((height, width, 3), dtype=tile_chw.dtype)

    return tile_to_uint8(rgb)


def save_upload(file: UploadFile, project_slug: str) -> Path:
    stem = slugify(Path(file.filename or "orthomosaic").stem)
    ext = Path(file.filename or "").suffix or ".tif"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    project_upload_dir = UPLOAD_ROOT / project_slug
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    target = project_upload_dir / f"{stem}_{ts}{ext}"

    contents = file.file.read()
    target.write_bytes(contents)
    return target


def ingest_orthomosaic(project_row: sqlite3.Row, ortho_name: str, ortho_path: Path) -> dict:
    project_slug = project_row["slug"]
    ortho_slug = slugify(Path(ortho_name).stem)
    created_at = now_iso()

    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO orthomosaics (project_id, original_name, stored_path, tile_dir, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                project_row["id"],
                ortho_name,
                str(ortho_path.relative_to(BASE_DIR)),
                "",
                created_at,
            ),
        )
        orthomosaic_id = cur.lastrowid

        # Keep ingest folders unique, even if the same orthomosaic name is uploaded many times.
        tile_dir_rel = Path("static") / project_slug / f"ortho_{orthomosaic_id:06d}_{ortho_slug}"
        conn.execute(
            "UPDATE orthomosaics SET tile_dir = ? WHERE id = ?",
            (str(tile_dir_rel).replace("\\", "/"), orthomosaic_id),
        )

    tile_dir = BASE_DIR / tile_dir_rel
    tile_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with rasterio.open(ortho_path) as src:
        for row in range(0, src.height, TILE_SIZE):
            for col in range(0, src.width, TILE_SIZE):
                width = min(TILE_SIZE, src.width - col)
                height = min(TILE_SIZE, src.height - row)
                window = Window(col, row, width, height)
                tile = src.read(window=window)

                if tile.size == 0 or np.all(tile == 0):
                    continue

                rgb = raster_to_rgb(tile)

                file_name = f"o{orthomosaic_id:06d}_{ortho_slug}_r{row:06d}_c{col:06d}_{width}x{height}.jpg"
                full_path = tile_dir / file_name
                Image.fromarray(rgb, mode="RGB").save(full_path, format="JPEG", quality=95)

                rel_path = "/" + str(full_path.relative_to(BASE_DIR)).replace("\\", "/")

                with db() as conn:
                    conn.execute(
                        """
                        INSERT INTO tiles (
                            project_id, orthomosaic_id, file_name, rel_path,
                            width, height, row_idx, col_idx, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_row["id"],
                            orthomosaic_id,
                            file_name,
                            rel_path,
                            int(width),
                            int(height),
                            int(row),
                            int(col),
                            now_iso(),
                        ),
                    )
                count += 1

    return {
        "orthomosaic_id": orthomosaic_id,
        "tiles_created": count,
        "tile_dir": "/" + str(tile_dir.relative_to(BASE_DIR)).replace("\\", "/"),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/hello")
def hello():
    return {
        "app": "DataBLaber",
        "message": "Orthomosaic labeling backend ready.",
        "version": "0.2.0",
    }


@app.get("/api/classes")
def list_classes():
    with db() as conn:
        rows = conn.execute("SELECT * FROM label_classes ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/classes")
def create_class(payload: ClassCreate):
    name = payload.name.strip()
    if not CLASS_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=400,
            detail="Class name must use only letters, numbers, underscore or dash (no spaces).",
        )

    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO label_classes (name, created_at) VALUES (?, ?)",
                (name, now_iso()),
            )
            class_id = cur.lastrowid
            row = conn.execute("SELECT * FROM label_classes WHERE id = ?", (class_id,)).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Class already exists")

    return dict(row)


@app.delete("/api/classes/{class_id}")
def delete_class(class_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM label_classes WHERE id = ?", (class_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Class not found")

        used = conn.execute("SELECT COUNT(*) as c FROM annotations WHERE class_id = ?", (class_id,)).fetchone()["c"]
        if used > 0:
            raise HTTPException(status_code=400, detail="Class is in use by annotations")

        conn.execute("DELETE FROM label_classes WHERE id = ?", (class_id,))
    return {"deleted": class_id}


@app.get("/api/projects")
def list_projects():
    with db() as conn:
        rows = conn.execute(
            """
            SELECT p.*, COUNT(t.id) as tile_count
            FROM projects p
            LEFT JOIN tiles t ON t.project_id = p.id
            GROUP BY p.id
            ORDER BY p.id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/projects")
def create_project(payload: ProjectCreate):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    slug = slugify(name)

    with db() as conn:
        existing = conn.execute("SELECT id FROM projects WHERE slug = ?", (slug,)).fetchone()
        if existing:
            suffix = 2
            while conn.execute("SELECT id FROM projects WHERE slug = ?", (f"{slug}_{suffix}",)).fetchone():
                suffix += 1
            slug = f"{slug}_{suffix}"

        try:
            cur = conn.execute(
                "INSERT INTO projects (name, slug, created_at) VALUES (?, ?, ?)",
                (name, slug, now_iso()),
            )
            project_id = cur.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Project name already exists")

        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

    (STATIC_ROOT / slug).mkdir(parents=True, exist_ok=True)
    return dict(row)


@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    project = ensure_project(project_id)
    with db() as conn:
        mosaics = conn.execute(
            "SELECT * FROM orthomosaics WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()
        tiles = conn.execute(
            "SELECT * FROM tiles WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()

    return {
        "project": dict(project),
        "orthomosaics": [dict(m) for m in mosaics],
        "tiles": [dict(t) for t in tiles],
    }


@app.post("/api/projects/{project_id}/ingest")
def ingest(project_id: int, orthomosaic: UploadFile = File(...)):
    project = ensure_project(project_id)
    upload_path = save_upload(orthomosaic, project["slug"])
    result = ingest_orthomosaic(project, orthomosaic.filename or upload_path.name, upload_path)
    return result


@app.get("/api/projects/{project_id}/tiles")
def list_project_tiles(project_id: int):
    ensure_project(project_id)
    with db() as conn:
        rows = conn.execute(
            """
            SELECT t.*,
                (SELECT COUNT(*) FROM annotations a WHERE a.tile_id = t.id) as annotation_count
            FROM tiles t
            WHERE t.project_id = ?
            ORDER BY t.id ASC
            """,
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/tiles/{tile_id}")
def get_tile(tile_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Tile not found")
    return dict(row)


@app.get("/api/tiles/{tile_id}/annotations")
def list_tile_annotations(tile_id: int):
    with db() as conn:
        exists = conn.execute("SELECT id FROM tiles WHERE id = ?", (tile_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Tile not found")

        rows = conn.execute(
            """
            SELECT a.*, c.name as class_name
            FROM annotations a
            JOIN label_classes c ON c.id = a.class_id
            WHERE a.tile_id = ?
            ORDER BY a.id ASC
            """,
            (tile_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/tiles/{tile_id}/annotations")
def create_annotation(tile_id: int, payload: AnnotationCreate):
    values = [payload.x_center, payload.y_center, payload.width, payload.height]
    if any(v < 0 or v > 1 for v in values):
        raise HTTPException(status_code=400, detail="Annotation values must be normalized between 0 and 1")
    if payload.width <= 0 or payload.height <= 0:
        raise HTTPException(status_code=400, detail="Annotation width/height must be > 0")

    with db() as conn:
        tile = conn.execute("SELECT id FROM tiles WHERE id = ?", (tile_id,)).fetchone()
        if not tile:
            raise HTTPException(status_code=404, detail="Tile not found")

        cls = conn.execute("SELECT id FROM label_classes WHERE id = ?", (payload.class_id,)).fetchone()
        if not cls:
            raise HTTPException(status_code=400, detail="Class does not exist")

        cur = conn.execute(
            """
            INSERT INTO annotations (tile_id, class_id, x_center, y_center, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tile_id,
                payload.class_id,
                payload.x_center,
                payload.y_center,
                payload.width,
                payload.height,
                now_iso(),
            ),
        )
        ann_id = cur.lastrowid
        row = conn.execute(
            """
            SELECT a.*, c.name as class_name
            FROM annotations a
            JOIN label_classes c ON c.id = a.class_id
            WHERE a.id = ?
            """,
            (ann_id,),
        ).fetchone()

    return dict(row)


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    with db() as conn:
        row = conn.execute("SELECT id FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Annotation not found")
        conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    return {"deleted": annotation_id}


@app.put("/api/annotations/{annotation_id}")
def update_annotation(annotation_id: int, payload: AnnotationUpdate):
    fields = ["x_center", "y_center", "width", "height"]
    incoming = payload.model_dump(exclude_unset=True)
    if not incoming:
        raise HTTPException(status_code=400, detail="No annotation fields to update")

    if "class_id" in incoming:
        with db() as conn:
            cls = conn.execute("SELECT id FROM label_classes WHERE id = ?", (incoming["class_id"],)).fetchone()
        if not cls:
            raise HTTPException(status_code=400, detail="Class does not exist")

    for key in fields:
        if key in incoming:
            value = incoming[key]
            if value is None or value < 0 or value > 1:
                raise HTTPException(status_code=400, detail=f"{key} must be normalized between 0 and 1")

    if "width" in incoming and incoming["width"] <= 0:
        raise HTTPException(status_code=400, detail="width must be > 0")
    if "height" in incoming and incoming["height"] <= 0:
        raise HTTPException(status_code=400, detail="height must be > 0")

    with db() as conn:
        existing = conn.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Annotation not found")

        merged = dict(existing)
        merged.update(incoming)
        if merged["width"] <= 0 or merged["height"] <= 0:
            raise HTTPException(status_code=400, detail="Annotation width/height must be > 0")
        if any(merged[k] < 0 or merged[k] > 1 for k in fields):
            raise HTTPException(status_code=400, detail="Annotation values must be normalized between 0 and 1")

        assignments = []
        values = []
        for key in ["class_id", "x_center", "y_center", "width", "height"]:
            if key in incoming:
                assignments.append(f"{key} = ?")
                values.append(incoming[key])
        values.append(annotation_id)

        conn.execute(f"UPDATE annotations SET {', '.join(assignments)} WHERE id = ?", values)
        row = conn.execute(
            """
            SELECT a.*, c.name as class_name
            FROM annotations a
            JOIN label_classes c ON c.id = a.class_id
            WHERE a.id = ?
            """,
            (annotation_id,),
        ).fetchone()

    return dict(row)


def yolo_lines_for_tile(conn: sqlite3.Connection, tile_id: int, class_index_map: dict) -> List[str]:
    anns = conn.execute(
        "SELECT class_id, x_center, y_center, width, height FROM annotations WHERE tile_id = ? ORDER BY id ASC",
        (tile_id,),
    ).fetchall()
    lines = []
    for ann in anns:
        class_idx = class_index_map.get(ann["class_id"])
        if class_idx is None:
            continue
        lines.append(
            f"{class_idx} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
        )
    return lines


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: int):
    project = ensure_project(project_id)

    with db() as conn:
        tiles = conn.execute("SELECT * FROM tiles WHERE project_id = ? ORDER BY id ASC", (project_id,)).fetchall()
        classes = conn.execute(
            """
            SELECT DISTINCT c.id, c.name
            FROM label_classes c
            JOIN annotations a ON a.class_id = c.id
            JOIN tiles t ON t.id = a.tile_id
            WHERE t.project_id = ?
            ORDER BY c.id ASC
            """,
            (project_id,),
        ).fetchall()

        class_index_map = {row["id"]: idx for idx, row in enumerate(classes)}
        class_names = [row["name"] for row in classes]

        temp_dir = Path(tempfile.mkdtemp(prefix="databird_export_"))
        export_root = temp_dir / f"{project['slug']}_yolo"
        labels_dir = export_root / "labels"
        images_dir = export_root / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        for tile in tiles:
            src = BASE_DIR / tile["rel_path"].lstrip("/")
            if not src.exists():
                continue

            export_stem = f"tile_{tile['id']:08d}"
            img_target = images_dir / f"{export_stem}.jpg"
            img_target.write_bytes(src.read_bytes())

            txt_name = f"{export_stem}.txt"
            label_target = labels_dir / txt_name
            lines = yolo_lines_for_tile(conn, tile["id"], class_index_map)
            label_target.write_text("\n".join(lines), encoding="utf-8")

        names_path = export_root / "classes.txt"
        names_path.write_text("\n".join(class_names), encoding="utf-8")

        data_yaml = (
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        (export_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

        zip_path = temp_dir / f"{project['slug']}_yolo.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in export_root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=str(file_path.relative_to(export_root)))

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{project['slug']}_yolo.zip",
    )
