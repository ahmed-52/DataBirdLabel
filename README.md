# DataBLaber (Bird Labeling MVP)

Simple orthomosaic labeling tool with FastAPI + React. YOLO dataset export

## What it does

- Define persistent label classes (`bird_name`, no spaces)
- Create projects (each gets a folder under `static/<project_slug>`)
- Ingest orthomosaic files and tile into `1280 x 1280` JPG tiles
- Label tiles with bounding boxes in the UI
- Store annotations in SQLite
- Export YOLO dataset zip:
  - `images/*.jpg`
  - `labels/*.txt`
  - `classes.txt`
  - `data.yaml`

## Structure

- `backend/app/main.py` - API, DB schema, tiling, export
- `backend/requirements.txt`
- `frontend/src/App.jsx` - UI workflow and labeling canvas
- `static/` - generated project tile folders
- `data/databirdlabel.db` - SQLite database (auto-created)

## Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`, backend on `http://localhost:8000`.

## Labeling shortcuts

- `D` toggles draw mode
- `←` and `→` move across tiles

## Notes

- Tiling is implemented with `rasterio` and follows window-based slicing logic.
- Empty tiles (`all zeros`) are skipped.
- Edge tiles smaller than `1280` are preserved and still exported in YOLO format.
