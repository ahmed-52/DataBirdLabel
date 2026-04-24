# DataBirdLabel

Collaborative orthomosaic labeling tool for bird population surveys. Frontend-only React app on Firebase Hosting, with Firestore for annotations and Firebase Storage for tile images.

Live at: **https://databirdlabel.web.app**

## Architecture

- **Frontend**: React + Vite, deployed to Firebase Hosting
- **Database**: Firestore (annotations, classes, projects, tiles, sessions)
- **Images**: Firebase Storage (tile JPGs from orthomosaics)
- **Admin CLI**: Local Python script for ingesting GeoTIFFs and managing classes/projects

Labelers use the web app. The admin (you) manages classes, projects, and data export from the command line.

## Setup (admin only)

### Prerequisites

```bash
pip install -r backend/requirements.txt
gcloud auth application-default login --project databirdlabel
```

### Environment

Create `frontend/.env`:

```
VITE_FIREBASE_API_KEY=<your-api-key>
VITE_FIREBASE_AUTH_DOMAIN=databirdlabel.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=databirdlabel
VITE_FIREBASE_STORAGE_BUCKET=databirdlabel.firebasestorage.app
VITE_FIREBASE_APP_ID=<your-app-id>
VITE_GATE_KEYWORD=<your-keyword>
```

## Admin CLI

All commands run from `backend/`:

```bash
cd backend
source venv/bin/activate
export GOOGLE_CLOUD_PROJECT=databirdlabel
```

### Add species classes

```bash
python -m app.main add-class asian_openbill
python -m app.main add-class white_birds
python -m app.main add-class black_birds
python -m app.main add-class black_headed_ibis
```

### List classes

```bash
python -m app.main list-classes
```

### Add labelers (users)

The gate screen shows a dropdown of registered labelers; their name is stamped on every annotation they create, edit, or delete.

```bash
python -m app.main add-user "Chea Monysocheata"
python -m app.main add-user "Rob Tizard"
python -m app.main list-users
```

### Ingest a GeoTIFF (creates a project + tiles)

```bash
python -m app.main ingest "ARU1" /path/to/orthomosaic.tif
```

This tiles the GeoTIFF into 1280x1280 JPGs, uploads them to Firebase Storage, and creates the project + tile documents in Firestore. One orthomosaic per project.

### List projects

```bash
python -m app.main list-projects
```

### Export annotations (YOLO format)

```bash
python -m app.main export 1 /path/to/output/
```

Exports a zip with:
- `images/*.jpg` — tile images
- `labels/*.txt` — YOLO annotation files
- `classes.txt` — class names
- `data.yaml` — YOLO dataset config
- `report.txt` — export summary (counts: total exported, unsure, soft-deleted skipped)

Soft-deleted annotations are kept in Firestore (for audit / potential undo) but excluded from the export. Annotations flagged "unsure" (with a secondary species) are exported using their primary class only — YOLO is one-class-per-box.

## Deploying updates

```bash
cd frontend && npm run build && cd ..
firebase deploy --only hosting --project databirdlabel
```

## For labelers

1. Go to https://databirdlabel.web.app
2. Pick your name from the dropdown and enter the keyword
3. Pick a project, click **Label**
4. Use **D** to draw boxes, **1–4** to switch active class, **+/-** to zoom, arrow keys to navigate tiles
5. Click a class on the left to set what you're labeling
6. If you aren't sure, pick your best guess as the primary class, then use the "Possible 2nd species (unsure)" dropdown in the sidebar to record your second guess — both will be shown on the box as `primary / ?secondary`
7. Only one person can label at a time (2-minute idle timeout)

Every box you create, edit, or delete is tagged with your name. Deleted boxes are kept in the database for audit (they disappear from the UI but aren't really gone).

## Session lock

One labeler at a time. The app sends a heartbeat every 30 seconds. If you close the tab or go idle for 2 minutes, the lock releases and someone else can start.

## Security

- Gate keyword is a lightweight deterrent, not real auth
- Firestore rules block web writes to classes, projects, tiles (admin CLI uses service account)
- Annotations and sessions are writable from the web app
- Firebase config keys are public identifiers, not secrets — Firestore rules control access
