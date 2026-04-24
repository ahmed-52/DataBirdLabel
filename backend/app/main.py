"""
Local-only CLI for managing DataBirdLabel on Firebase.

Usage:
    python -m app.main add-class <class_name>
    python -m app.main list-classes
    python -m app.main add-user "<full name>"
    python -m app.main list-users
    python -m app.main list-projects
    python -m app.main ingest <project_name> <geotiff_path>
    python -m app.main export <project_name> <output_dir>

Requires GOOGLE_CLOUD_PROJECT env var or gcloud default project.
"""

import io
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from .firebase_client import get_bucket, get_db, next_id

TILE_SIZE = 1280


def tile_to_uint8(tile: np.ndarray) -> np.ndarray:
    if tile.size == 0:
        return tile.astype(np.uint8)
    if tile.dtype == np.uint8:
        return tile
    tile_f = tile.astype(np.float32)
    t_min, t_max = float(tile_f.min()), float(tile_f.max())
    if t_max <= t_min:
        return np.zeros_like(tile_f, dtype=np.uint8)
    return (((tile_f - t_min) / (t_max - t_min)) * 255.0).clip(0, 255).astype(np.uint8)


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


def slugify(value: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "project"


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cmd_add_class(name: str):
    import re
    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    if not pattern.match(name):
        print(f"Error: class name must be alphanumeric/underscore/dash, got '{name}'")
        sys.exit(1)

    db = get_db()
    existing = db.collection("classes").where("name", "==", name).limit(1).get()
    if existing:
        print(f"Class '{name}' already exists (id={existing[0].get('id')})")
        return

    cid = next_id("classes")
    db.collection("classes").document(str(cid)).set({
        "id": cid,
        "name": name,
        "created_at": now_iso(),
    })
    print(f"Created class '{name}' (id={cid})")


def cmd_list_classes():
    db = get_db()
    docs = db.collection("classes").order_by("id").get()
    if not docs:
        print("No classes defined.")
        return
    for d in docs:
        print(f"  [{d.get('id')}] {d.get('name')}")


def cmd_add_user(name: str):
    name = name.strip()
    if not name:
        print("Error: user name cannot be empty")
        sys.exit(1)
    if len(name) > 80:
        print("Error: user name too long (max 80 chars)")
        sys.exit(1)

    db = get_db()
    existing = db.collection("users").where("name", "==", name).limit(1).get()
    if existing:
        print(f"User '{name}' already exists (id={existing[0].get('id')})")
        return

    uid = next_id("users")
    db.collection("users").document(str(uid)).set({
        "id": uid,
        "name": name,
        "created_at": now_iso(),
    })
    print(f"Created user '{name}' (id={uid})")


def cmd_list_users():
    db = get_db()
    docs = db.collection("users").order_by("id").get()
    if not docs:
        print("No users defined.")
        return
    for d in docs:
        print(f"  [{d.get('id')}] {d.get('name')}")


def cmd_list_projects():
    db = get_db()
    docs = db.collection("projects").order_by("id").get()
    if not docs:
        print("No projects.")
        return
    for d in docs:
        data = d.to_dict()
        tile_count = len(db.collection("tiles").where("project_id", "==", data["id"]).get())
        print(f"  [{data['id']}] {data['name']} ({tile_count} tiles)")


def cmd_ingest(project_name: str, geotiff_path: str):
    path = Path(geotiff_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    db = get_db()
    slug = slugify(project_name)

    # Check if project exists
    existing = db.collection("projects").where("slug", "==", slug).limit(1).get()
    if existing:
        print(f"Error: project '{project_name}' (slug={slug}) already exists")
        sys.exit(1)

    # Create project
    pid = next_id("projects")
    db.collection("projects").document(str(pid)).set({
        "id": pid,
        "name": project_name,
        "slug": slug,
        "created_at": now_iso(),
    })
    print(f"Created project '{project_name}' (id={pid})")

    # Create orthomosaic record
    oid = next_id("orthomosaics")
    db.collection("orthomosaics").document(str(oid)).set({
        "id": oid,
        "project_id": pid,
        "original_name": path.name,
        "created_at": now_iso(),
    })

    bucket = get_bucket()
    count = 0

    print(f"Tiling {path.name}...")
    with rasterio.open(path) as src:
        total_rows = (src.height + TILE_SIZE - 1) // TILE_SIZE
        total_cols = (src.width + TILE_SIZE - 1) // TILE_SIZE
        total = total_rows * total_cols
        print(f"  Raster: {src.width}x{src.height}, ~{total} potential tiles")

        for row in range(0, src.height, TILE_SIZE):
            for col in range(0, src.width, TILE_SIZE):
                width = min(TILE_SIZE, src.width - col)
                height = min(TILE_SIZE, src.height - row)
                window = Window(col, row, width, height)
                tile = src.read(window=window)

                if tile.size == 0 or np.all(tile == 0):
                    continue

                rgb = raster_to_rgb(tile)
                img = Image.fromarray(rgb, mode="RGB")

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                buf.seek(0)

                file_name = f"r{row:06d}_c{col:06d}_{width}x{height}.jpg"
                blob_path = f"tiles/{pid}/{oid}/{file_name}"

                blob = bucket.blob(blob_path)
                blob.upload_from_file(buf, content_type="image/jpeg")

                # Construct public URL (bucket must have public read via Storage rules)
                storage_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob_path.replace('/', '%2F')}?alt=media"

                tid = next_id("tiles")
                db.collection("tiles").document(str(tid)).set({
                    "id": tid,
                    "project_id": pid,
                    "orthomosaic_id": oid,
                    "file_name": file_name,
                    "storage_url": storage_url,
                    "width": int(width),
                    "height": int(height),
                    "row_idx": int(row),
                    "col_idx": int(col),
                    "created_at": now_iso(),
                })
                count += 1
                print(f"  [{count}] {file_name}", end="\r")

    # Update project with tile count
    db.collection("projects").document(str(pid)).update({"tile_count": count})
    print(f"\nDone. Uploaded {count} tiles to Firebase Storage.")


def cmd_export(project_name: str, output_dir: str):
    import zipfile as zf_mod

    db = get_db()

    # Look up by name or slug
    slug = slugify(project_name)
    results = db.collection("projects").where("slug", "==", slug).limit(1).get()
    if not results:
        # Try exact name match
        results = db.collection("projects").where("name", "==", project_name).limit(1).get()
    if not results:
        print(f"Error: project '{project_name}' not found")
        print("Available projects:")
        cmd_list_projects()
        sys.exit(1)
    proj_data = results[0].to_dict()
    pid = proj_data["id"]
    slug = proj_data.get("slug", f"project_{pid}")

    try:
        tiles = db.collection("tiles").where("project_id", "==", pid).order_by("id").get()
    except Exception:
        tiles = db.collection("tiles").where("project_id", "==", pid).get()
        tiles = sorted(tiles, key=lambda t: t.get("id"))
    if not tiles:
        print("No tiles in this project.")
        return

    # Get all annotations and classes used.
    # Soft-deleted rows are kept in Firestore for audit but excluded from exports.
    class_ids_used = set()
    all_anns = {}
    skipped_deleted = 0
    unsure_count = 0
    for t in tiles:
        td = t.to_dict()
        try:
            anns = db.collection("annotations").where("tile_id", "==", td["id"]).order_by("id").get()
        except Exception:
            anns = db.collection("annotations").where("tile_id", "==", td["id"]).get()
            anns = sorted(anns, key=lambda a: a.get("id"))
        raw = [a.to_dict() for a in anns]
        live = []
        for a in raw:
            if a.get("deleted") is True:
                skipped_deleted += 1
                continue
            if a.get("secondary_class_id"):
                unsure_count += 1
            live.append(a)
        all_anns[td["id"]] = live
        for a in live:
            class_ids_used.add(a["class_id"])

    classes = []
    for cid in sorted(class_ids_used):
        cdoc = db.collection("classes").document(str(cid)).get()
        if cdoc.exists:
            classes.append({"id": cid, "name": cdoc.get("name")})

    class_index_map = {c["id"]: idx for idx, c in enumerate(classes)}
    class_names = [c["name"] for c in classes]

    out_path = Path(output_dir) / f"{slug}_yolo.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bucket = get_bucket()
    print(f"Exporting {len(tiles)} tiles...")

    with zf_mod.ZipFile(out_path, "w", zf_mod.ZIP_DEFLATED) as zf:
        for i, t in enumerate(tiles):
            td = t.to_dict()
            stem = f"tile_{td['id']:08d}"

            # Download image
            blob_path = f"tiles/{pid}/{td['orthomosaic_id']}/{td['file_name']}"
            blob = bucket.blob(blob_path)
            if blob.exists():
                zf.writestr(f"images/{stem}.jpg", blob.download_as_bytes())

            # Labels
            lines = []
            skipped = 0
            for a in all_anns.get(td["id"], []):
                cidx = class_index_map.get(a["class_id"])
                if cidx is None:
                    skipped += 1
                    continue
                lines.append(f"{cidx} {a['x_center']:.6f} {a['y_center']:.6f} {a['width']:.6f} {a['height']:.6f}")
            if skipped:
                print(f"\n  WARNING: {skipped} annotations on {td['file_name']} reference deleted classes — skipped")
            zf.writestr(f"labels/{stem}.txt", "\n".join(lines))

            print(f"  [{i+1}/{len(tiles)}] {td['file_name']}", end="\r")

        import json
        zf.writestr("classes.txt", "\n".join(class_names))
        data_yaml = f"path: .\ntrain: images\nval: images\nnc: {len(class_names)}\nnames: {json.dumps(class_names)}\n"
        zf.writestr("data.yaml", data_yaml)

        exported_anns = sum(len(v) for v in all_anns.values())
        report = (
            f"project: {proj_data['name']} (id={pid})\n"
            f"exported_at: {now_iso()}\n"
            f"tiles: {len(tiles)}\n"
            f"classes: {len(class_names)}\n"
            f"annotations_exported: {exported_anns}\n"
            f"  of which unsure (had a secondary class, exported as primary only): {unsure_count}\n"
            f"annotations_skipped_soft_deleted: {skipped_deleted}\n"
        )
        zf.writestr("report.txt", report)

    print(
        f"\nExported to {out_path} "
        f"({len(class_names)} classes, {sum(len(v) for v in all_anns.values())} annotations, "
        f"{unsure_count} unsure, {skipped_deleted} soft-deleted skipped)"
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "add-class" and len(sys.argv) == 3:
        cmd_add_class(sys.argv[2])
    elif cmd == "list-classes":
        cmd_list_classes()
    elif cmd == "add-user" and len(sys.argv) == 3:
        cmd_add_user(sys.argv[2])
    elif cmd == "list-users":
        cmd_list_users()
    elif cmd == "list-projects":
        cmd_list_projects()
    elif cmd == "ingest" and len(sys.argv) == 4:
        cmd_ingest(sys.argv[2], sys.argv[3])
    elif cmd == "export" and len(sys.argv) == 4:
        cmd_export(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
