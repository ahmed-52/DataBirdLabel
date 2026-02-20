#!/usr/bin/env python3
"""
One-off local importer for the ARU projectx dataset.

What it does:
1) Creates a new project row.
2) Adds one orthomosaic row.
3) Tiles the TIFF into static/<project_slug>/...
4) Reads GeoJSON with geopandas, reprojects to raster CRS when needed.
5) Writes annotations in the app's normalized YOLO-style DB format.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

TILE_SIZE = 1280
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_ROOT = BASE_DIR / "static"
DATA_ROOT = BASE_DIR / "data"
DB_PATH = DATA_ROOT / "databirdlabel.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "project"


def normalize_class_name(value: str | None, fallback: str = "bird") -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (float, np.floating)) and np.isnan(value):
        text = ""
    else:
        text = str(value)
    text = text.strip().lower()
    if not text:
        text = fallback.strip().lower() or "bird"
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "bird"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
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


def ensure_class(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM label_classes WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO label_classes (name, created_at) VALUES (?, ?)",
        (name, now_iso()),
    )
    return int(cur.lastrowid)


def create_project(conn: sqlite3.Connection, project_name: str) -> sqlite3.Row:
    slug = slugify(project_name)
    existing = conn.execute("SELECT id FROM projects WHERE slug = ?", (slug,)).fetchone()
    if existing:
        suffix = 2
        while conn.execute("SELECT id FROM projects WHERE slug = ?", (f"{slug}_{suffix}",)).fetchone():
            suffix += 1
        slug = f"{slug}_{suffix}"
    cur = conn.execute(
        "INSERT INTO projects (name, slug, created_at) VALUES (?, ?, ?)",
        (project_name, slug, now_iso()),
    )
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (cur.lastrowid,)).fetchone()
    (STATIC_ROOT / project["slug"]).mkdir(parents=True, exist_ok=True)
    return project


def raster_to_rgb(tile_chw: np.ndarray) -> np.ndarray:
    bands, height, width = tile_chw.shape
    if bands >= 3:
        rgb = np.moveaxis(tile_chw[:3], 0, -1)
    elif bands == 1:
        mono = tile_chw[0]
        rgb = np.stack([mono, mono, mono], axis=-1)
    else:
        rgb = np.zeros((height, width, 3), dtype=tile_chw.dtype)

    if rgb.dtype != np.uint8:
        tile_f = rgb.astype(np.float32)
        t_min = float(tile_f.min())
        t_max = float(tile_f.max())
        if t_max <= t_min:
            rgb = np.zeros_like(tile_f, dtype=np.uint8)
        else:
            scaled = (tile_f - t_min) / (t_max - t_min)
            rgb = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def bounds_to_local_pixels(
    transform: rasterio.Affine,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    tile_col: int,
    tile_row: int,
) -> tuple[float, float, float, float]:
    inv = ~transform
    corners = [
        inv * (minx, miny),
        inv * (minx, maxy),
        inv * (maxx, miny),
        inv * (maxx, maxy),
    ]
    cols = [c[0] for c in corners]
    rows = [c[1] for c in corners]
    left = min(cols) - tile_col
    right = max(cols) - tile_col
    top = min(rows) - tile_row
    bottom = max(rows) - tile_row
    return left, top, right, bottom


def main() -> None:
    parser = argparse.ArgumentParser(description="Import ARU projectx data into DataBirdLabel DB.")
    parser.add_argument("--project-name", default="ARU1 r025")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "backend" / "projectx"))
    parser.add_argument("--tif", default="ARU1_r025_clipped_ortho.tif")
    parser.add_argument("--geojson", default="ARU1_r025_labels.geojson")
    parser.add_argument("--default-class", default="bird")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    tif_path = dataset_dir / args.tif
    geojson_path = dataset_dir / args.geojson

    if not tif_path.exists():
        raise FileNotFoundError(f"Missing TIFF: {tif_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"Missing GeoJSON: {geojson_path}")

    init_db()

    with db() as conn:
        project = create_project(conn, args.project_name)
        default_class = normalize_class_name(args.default_class)
        class_cache: dict[str, int] = {default_class: ensure_class(conn, default_class)}

        cur = conn.execute(
            """
            INSERT INTO orthomosaics (project_id, original_name, stored_path, tile_dir, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(project["id"]),
                tif_path.name,
                str(tif_path.relative_to(BASE_DIR)).replace("\\", "/"),
                "",
                now_iso(),
            ),
        )
        orthomosaic_id = int(cur.lastrowid)
        ortho_slug = slugify(tif_path.stem)
        tile_dir_rel = Path("static") / project["slug"] / f"ortho_{orthomosaic_id:06d}_{ortho_slug}"
        conn.execute(
            "UPDATE orthomosaics SET tile_dir = ? WHERE id = ?",
            (str(tile_dir_rel).replace("\\", "/"), orthomosaic_id),
        )
        tile_dir_abs = BASE_DIR / tile_dir_rel
        tile_dir_abs.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise RuntimeError("GeoJSON has zero features.")
        if gdf.crs is None:
            raise RuntimeError("GeoJSON CRS is missing.")
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
        if gdf.empty:
            raise RuntimeError("GeoJSON has no valid geometries.")

        tiles_created = 0
        annotations_created = 0

        with rasterio.open(tif_path) as src:
            if src.crs is None:
                raise RuntimeError("Orthomosaic CRS is missing.")
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            sindex = gdf.sindex

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
                    file_path = tile_dir_abs / file_name
                    Image.fromarray(rgb, mode="RGB").save(file_path, format="JPEG", quality=95)
                    rel_path = "/" + str(file_path.relative_to(BASE_DIR)).replace("\\", "/")

                    tile_cur = conn.execute(
                        """
                        INSERT INTO tiles (
                            project_id, orthomosaic_id, file_name, rel_path,
                            width, height, row_idx, col_idx, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(project["id"]),
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
                    tile_id = int(tile_cur.lastrowid)
                    tiles_created += 1

                    tile_geom = box(*window_bounds(window, src.transform))
                    candidate_ids = list(sindex.intersection(tile_geom.bounds))
                    if not candidate_ids:
                        continue

                    for feature in gdf.iloc[candidate_ids].itertuples():
                        geom = feature.geometry
                        if geom is None or geom.is_empty or not geom.intersects(tile_geom):
                            continue

                        clipped = geom.intersection(tile_geom)
                        if clipped.is_empty:
                            continue

                        minx, miny, maxx, maxy = clipped.bounds
                        left, top, right, bottom = bounds_to_local_pixels(
                            src.transform, minx, miny, maxx, maxy, tile_col=col, tile_row=row
                        )
                        left = max(0.0, min(float(width), left))
                        right = max(0.0, min(float(width), right))
                        top = max(0.0, min(float(height), top))
                        bottom = max(0.0, min(float(height), bottom))

                        if right <= left or bottom <= top:
                            if clipped.geom_type in {"Point", "MultiPoint"}:
                                cx = min(max(left, 0.0), float(width))
                                cy = min(max(top, 0.0), float(height))
                                left = max(0.0, cx - 0.5)
                                top = max(0.0, cy - 0.5)
                                right = min(float(width), left + 1.0)
                                bottom = min(float(height), top + 1.0)
                            else:
                                continue

                        x_center = ((left + right) / 2.0) / float(width)
                        y_center = ((top + bottom) / 2.0) / float(height)
                        w_norm = (right - left) / float(width)
                        h_norm = (bottom - top) / float(height)
                        if w_norm <= 0.0 or h_norm <= 0.0:
                            continue

                        species = getattr(feature, "species", None)
                        class_name = normalize_class_name(species, fallback=default_class)
                        class_id = class_cache.get(class_name)
                        if class_id is None:
                            class_id = ensure_class(conn, class_name)
                            class_cache[class_name] = class_id

                        conn.execute(
                            """
                            INSERT INTO annotations (tile_id, class_id, x_center, y_center, width, height, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                tile_id,
                                class_id,
                                min(max(x_center, 0.0), 1.0),
                                min(max(y_center, 0.0), 1.0),
                                min(max(w_norm, 0.0), 1.0),
                                min(max(h_norm, 0.0), 1.0),
                                now_iso(),
                            ),
                        )
                        annotations_created += 1

        print(f"Project created: id={project['id']} slug={project['slug']}")
        print(f"Orthomosaic id: {orthomosaic_id}")
        print(f"Tiles created: {tiles_created}")
        print(f"Annotations created: {annotations_created}")
        print(f"Tile dir: /{str(tile_dir_rel).replace('\\', '/')}")


if __name__ == "__main__":
    main()
