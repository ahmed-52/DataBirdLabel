import { useEffect, useMemo, useRef, useState } from "react"

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || "http://localhost:8000"

const COLOR_PALETTE = [
  "#22c55e",
  "#f97316",
  "#06b6d4",
  "#eab308",
  "#ef4444",
  "#3b82f6",
  "#84cc16",
  "#ec4899",
  "#14b8a6",
  "#a855f7",
  "#10b981",
  "#f43f5e",
]

function colorForClassId(classId) {
  const id = Number(classId) || 0
  const hashed = (id * 2654435761) >>> 0
  return COLOR_PALETTE[hashed % COLOR_PALETTE.length]
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function hexToRgba(hex, alpha) {
  const c = hex.replace("#", "")
  const n = Number.parseInt(c, 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

async function api(path, options = {}) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = "Request failed"
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }

  const contentType = res.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    return res.json()
  }
  return res
}

function toNormRect(rect, containerW, containerH) {
  const x1 = Math.min(rect.x1, rect.x2)
  const y1 = Math.min(rect.y1, rect.y2)
  const x2 = Math.max(rect.x1, rect.x2)
  const y2 = Math.max(rect.y1, rect.y2)

  const w = x2 - x1
  const h = y2 - y1
  if (w < 4 || h < 4) return null

  return {
    x_center: (x1 + w / 2) / containerW,
    y_center: (y1 + h / 2) / containerH,
    width: w / containerW,
    height: h / containerH,
  }
}

function fromNormRect(ann, containerW, containerH) {
  const w = ann.width * containerW
  const h = ann.height * containerH
  const x = ann.x_center * containerW - w / 2
  const y = ann.y_center * containerH - h / 2
  return { x, y, w, h }
}

function normalizePixelRect(pixelRect, containerW, containerH) {
  const x = clamp(pixelRect.x, 0, containerW)
  const y = clamp(pixelRect.y, 0, containerH)
  const w = clamp(pixelRect.w, 0, containerW - x)
  const h = clamp(pixelRect.h, 0, containerH - y)
  return {
    x_center: (x + w / 2) / containerW,
    y_center: (y + h / 2) / containerH,
    width: w / containerW,
    height: h / containerH,
  }
}

export default function App() {
  const [tab, setTab] = useState("classes")
  const [classes, setClasses] = useState([])
  const [projects, setProjects] = useState([])

  const [newClass, setNewClass] = useState("")
  const [newProject, setNewProject] = useState("")

  const [selectedProjectId, setSelectedProjectId] = useState(null)
  const [projectTiles, setProjectTiles] = useState([])
  const [projectMosaics, setProjectMosaics] = useState([])

  const [selectedTileIndex, setSelectedTileIndex] = useState(0)
  const [tileAnnotations, setTileAnnotations] = useState([])

  const [selectedClassId, setSelectedClassId] = useState(null)
  const [selectedAnnotationId, setSelectedAnnotationId] = useState(null)
  const [drawMode, setDrawMode] = useState(false)
  const [dragRect, setDragRect] = useState(null)
  const [activeInteraction, setActiveInteraction] = useState(null)
  const [editingAnnotation, setEditingAnnotation] = useState(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const stageRef = useRef(null)

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) || null,
    [projects, selectedProjectId],
  )

  const selectedTile = projectTiles[selectedTileIndex] || null

  const displayedAnnotations = useMemo(() => {
    return tileAnnotations.map((ann) => {
      if (editingAnnotation && editingAnnotation.id === ann.id) {
        return { ...ann, ...editingAnnotation }
      }
      return ann
    })
  }, [tileAnnotations, editingAnnotation])

  function getStagePoint(e) {
    if (!stageRef.current) return null
    const r = stageRef.current.getBoundingClientRect()
    return {
      x: clamp(e.clientX - r.left, 0, r.width),
      y: clamp(e.clientY - r.top, 0, r.height),
      w: r.width,
      h: r.height,
    }
  }

  function findAnnotationAtPoint(x, y, stageW, stageH) {
    for (let i = displayedAnnotations.length - 1; i >= 0; i -= 1) {
      const ann = displayedAnnotations[i]
      const r = fromNormRect(ann, stageW, stageH)
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
        return { ann, rect: r }
      }
    }
    return null
  }

  async function refreshClasses() {
    setClasses(await api("/api/classes"))
  }

  async function refreshProjects() {
    const rows = await api("/api/projects")
    setProjects(rows)
    if (!selectedProjectId && rows.length) {
      setSelectedProjectId(rows[0].id)
    }
  }

  async function refreshProjectDetails(projectId, { resetIndex = false } = {}) {
    if (!projectId) return
    const data = await api(`/api/projects/${projectId}`)
    setProjectMosaics(data.orthomosaics || [])
    setProjectTiles(data.tiles || [])
    if (resetIndex) setSelectedTileIndex(0)
  }

  async function refreshTileAnnotations(tileId) {
    if (!tileId) {
      setTileAnnotations([])
      return
    }
    const rows = await api(`/api/tiles/${tileId}/annotations`)
    setTileAnnotations(rows)
  }

  useEffect(() => {
    refreshClasses().catch((e) => setError(e.message))
    refreshProjects().catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (selectedProjectId) {
      refreshProjectDetails(selectedProjectId, { resetIndex: true }).catch((e) => setError(e.message))
    }
  }, [selectedProjectId])

  useEffect(() => {
    if (selectedTile) {
      refreshTileAnnotations(selectedTile.id).catch((e) => setError(e.message))
    } else {
      setTileAnnotations([])
    }
    setSelectedAnnotationId(null)
    setEditingAnnotation(null)
    setActiveInteraction(null)
  }, [selectedTile?.id])

  useEffect(() => {
    if (!selectedClassId && classes.length) {
      setSelectedClassId(classes[0].id)
    }
  }, [classes, selectedClassId])

  useEffect(() => {
    if (selectedTileIndex > Math.max(0, projectTiles.length - 1)) {
      setSelectedTileIndex(Math.max(0, projectTiles.length - 1))
    }
  }, [projectTiles.length, selectedTileIndex])

  useEffect(() => {
    function onKey(e) {
      const targetTag = (e.target?.tagName || "").toLowerCase()
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") {
        return
      }

      if (e.key.toLowerCase() === "d") {
        setDrawMode((d) => !d)
      }
      if (e.key === "ArrowRight") {
        setSelectedTileIndex((i) => Math.min(i + 1, Math.max(0, projectTiles.length - 1)))
      }
      if (e.key === "ArrowLeft") {
        setSelectedTileIndex((i) => Math.max(i - 1, 0))
      }
      if (!drawMode && selectedAnnotationId && (e.key === "Delete" || e.key === "Backspace")) {
        e.preventDefault()
        deleteAnnotation(selectedAnnotationId)
      }
    }

    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [projectTiles.length, drawMode, selectedAnnotationId])

  async function createClass() {
    if (!newClass.trim()) return
    setLoading(true)
    setError("")
    try {
      await api("/api/classes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newClass.trim() }),
      })
      setNewClass("")
      await refreshClasses()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function removeClass(classId) {
    setLoading(true)
    setError("")
    try {
      await api(`/api/classes/${classId}`, { method: "DELETE" })
      await refreshClasses()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function createProject() {
    if (!newProject.trim()) return
    setLoading(true)
    setError("")
    try {
      const p = await api("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newProject.trim() }),
      })
      setNewProject("")
      await refreshProjects()
      setSelectedProjectId(p.id)
      setTab("projects")
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function ingestOrthomosaic(file) {
    if (!file || !selectedProject) return
    setLoading(true)
    setError("")
    try {
      const fd = new FormData()
      fd.append("orthomosaic", file)
      await api(`/api/projects/${selectedProject.id}/ingest`, {
        method: "POST",
        body: fd,
      })
      await refreshProjects()
      await refreshProjectDetails(selectedProject.id, { resetIndex: false })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function exportProject() {
    if (!selectedProject) return
    setLoading(true)
    setError("")
    try {
      const res = await fetch(`/api/projects/${selectedProject.id}/export`)
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || "Export failed")
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${selectedProject.slug}_yolo.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function submitRect(rect) {
    if (!selectedTile || !selectedClassId || !stageRef.current) return
    const box = stageRef.current.getBoundingClientRect()
    const normalized = toNormRect(rect, box.width, box.height)
    if (!normalized) return

    setLoading(true)
    setError("")
    try {
      await api(`/api/tiles/${selectedTile.id}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: selectedClassId, ...normalized }),
      })
      await refreshTileAnnotations(selectedTile.id)
      await refreshProjects()
      await refreshProjectDetails(selectedProject.id, { resetIndex: false })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function updateAnnotation(annotationId, updates) {
    setLoading(true)
    setError("")
    try {
      await api(`/api/annotations/${annotationId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      })
      if (selectedTile) {
        await refreshTileAnnotations(selectedTile.id)
      }
      await refreshProjects()
      if (selectedProject) {
        await refreshProjectDetails(selectedProject.id, { resetIndex: false })
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function deleteAnnotation(annotationId) {
    if (!selectedTile) return
    setLoading(true)
    setError("")
    try {
      await api(`/api/annotations/${annotationId}`, { method: "DELETE" })
      await refreshTileAnnotations(selectedTile.id)
      await refreshProjects()
      await refreshProjectDetails(selectedProject.id, { resetIndex: false })
      if (selectedAnnotationId === annotationId) {
        setSelectedAnnotationId(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function onStageDown(e) {
    const p = getStagePoint(e)
    if (!p) return

    if (drawMode) {
      setDragRect({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
      return
    }

    const hit = findAnnotationAtPoint(p.x, p.y, p.w, p.h)
    if (!hit) {
      setSelectedAnnotationId(null)
      return
    }

    const isSelected = selectedAnnotationId === hit.ann.id
    const handleSize = 12
    const onResizeHandle =
      isSelected &&
      p.x >= hit.rect.x + hit.rect.w - handleSize &&
      p.x <= hit.rect.x + hit.rect.w + handleSize &&
      p.y >= hit.rect.y + hit.rect.h - handleSize &&
      p.y <= hit.rect.y + hit.rect.h + handleSize

    setSelectedAnnotationId(hit.ann.id)
    setSelectedClassId(hit.ann.class_id)

    if (onResizeHandle) {
      setActiveInteraction({
        type: "resize",
        annotationId: hit.ann.id,
        originRect: hit.rect,
      })
      return
    }

    setActiveInteraction({
      type: "move",
      annotationId: hit.ann.id,
      offsetX: p.x - hit.rect.x,
      offsetY: p.y - hit.rect.y,
      originRect: hit.rect,
    })
  }

  function onStageMove(e) {
    const p = getStagePoint(e)
    if (!p) return

    if (drawMode && dragRect) {
      setDragRect((prev) => (prev ? { ...prev, x2: p.x, y2: p.y } : prev))
      return
    }

    if (!activeInteraction) return

    const minSize = 6
    if (activeInteraction.type === "move") {
      const w = activeInteraction.originRect.w
      const h = activeInteraction.originRect.h
      const x = clamp(p.x - activeInteraction.offsetX, 0, p.w - w)
      const y = clamp(p.y - activeInteraction.offsetY, 0, p.h - h)
      const normalized = normalizePixelRect({ x, y, w, h }, p.w, p.h)
      setEditingAnnotation({ id: activeInteraction.annotationId, ...normalized })
      return
    }

    if (activeInteraction.type === "resize") {
      const x = activeInteraction.originRect.x
      const y = activeInteraction.originRect.y
      const w = clamp(p.x - x, minSize, p.w - x)
      const h = clamp(p.y - y, minSize, p.h - y)
      const normalized = normalizePixelRect({ x, y, w, h }, p.w, p.h)
      setEditingAnnotation({ id: activeInteraction.annotationId, ...normalized })
    }
  }

  async function onStageUp() {
    if (drawMode && dragRect) {
      const finalized = dragRect
      setDragRect(null)
      setDrawMode(false)
      await submitRect(finalized)
      return
    }

    if (!activeInteraction) return

    const annId = activeInteraction.annotationId
    const updates = editingAnnotation
      ? {
          x_center: editingAnnotation.x_center,
          y_center: editingAnnotation.y_center,
          width: editingAnnotation.width,
          height: editingAnnotation.height,
        }
      : null

    setActiveInteraction(null)
    setEditingAnnotation(null)

    if (updates) {
      await updateAnnotation(annId, updates)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      <aside
        className={`${sidebarOpen ? "w-80" : "w-14"} border-r border-zinc-200 bg-white p-3 flex flex-col gap-3 transition-all duration-200`}
      >
        <div className="flex items-center justify-between p-2 rounded-sm border border-zinc-200">
          {sidebarOpen && (
            <div className="flex items-center gap-2">
              <img src="/databird.png" alt="DataBLaber" className="h-8 w-8 object-contain" />
              <div>
                <h1 className="text-sm font-bold tracking-wide font-display">DataBLaber</h1>
                <p className="text-[10px] uppercase tracking-wider text-zinc-500 font-mono">Bird Label Tool</p>
              </div>
            </div>
          )}
          <button className="text-xs px-2 py-1 border rounded" onClick={() => setSidebarOpen((v) => !v)}>
            {sidebarOpen ? "Close" : "Open"}
          </button>
        </div>

        {sidebarOpen && (
          <>
            <button
              className={`w-full text-left px-3 py-2 rounded-sm border text-xs font-mono uppercase tracking-wide ${
                tab === "classes" ? "border-zinc-300 bg-zinc-50" : "border-transparent"
              }`}
              onClick={() => setTab("classes")}
            >
              Classes
            </button>
            <button
              className={`w-full text-left px-3 py-2 rounded-sm border text-xs font-mono uppercase tracking-wide ${
                tab === "projects" ? "border-zinc-300 bg-zinc-50" : "border-transparent"
              }`}
              onClick={() => setTab("projects")}
            >
              Projects
            </button>
            <button
              className={`w-full text-left px-3 py-2 rounded-sm border text-xs font-mono uppercase tracking-wide ${
                tab === "labeler" ? "border-zinc-300 bg-zinc-50" : "border-transparent"
              }`}
              onClick={() => setTab("labeler")}
              disabled={!selectedProject || projectTiles.length === 0}
            >
              Labeling
            </button>

            {tab === "labeler" && (
              <div className="border rounded bg-zinc-50 p-3 space-y-3">
                <div className="text-sm font-medium">Label Settings</div>
                <div className="text-xs text-zinc-600">
                  {selectedProject ? `Project: ${selectedProject.name}` : "No project selected"}
                </div>
                <div className="text-xs text-zinc-600">
                  Tile {projectTiles.length ? selectedTileIndex + 1 : 0} / {projectTiles.length}
                </div>

                <select
                  className="w-full border rounded p-2"
                  value={selectedClassId || ""}
                  onChange={(e) => setSelectedClassId(Number(e.target.value))}
                >
                  {classes.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>

                <button
                  className={`w-full px-3 py-2 rounded border ${drawMode ? "bg-zinc-900 text-white" : "bg-white"}`}
                  onClick={() => setDrawMode((v) => !v)}
                >
                  Draw mode: {drawMode ? "ON" : "OFF"}
                </button>

                <div className="text-xs text-zinc-600">Press D for one annotation, draw, it auto exits draw mode.</div>

                <div className="flex gap-2">
                  <button className="flex-1 border rounded px-2 py-1" onClick={() => setSelectedTileIndex((i) => Math.max(i - 1, 0))}>
                    Prev
                  </button>
                  <button
                    className="flex-1 border rounded px-2 py-1"
                    onClick={() => setSelectedTileIndex((i) => Math.min(i + 1, projectTiles.length - 1))}
                  >
                    Next
                  </button>
                </div>

                <div className="border-t pt-2 space-y-2">
                  <div className="text-sm font-medium">Annotations ({tileAnnotations.length})</div>
                  <div className="max-h-56 overflow-auto space-y-1">
                    {tileAnnotations.map((ann) => (
                      <button
                        key={ann.id}
                        className={`w-full text-left text-xs border rounded p-2 flex items-center justify-between bg-white ${
                          selectedAnnotationId === ann.id ? "ring-1 ring-zinc-500" : ""
                        }`}
                        onClick={() => setSelectedAnnotationId(ann.id)}
                      >
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: colorForClassId(ann.class_id) }}
                          />
                          {ann.class_name}
                        </span>
                        <span
                          className="text-red-700"
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            deleteAnnotation(ann.id)
                          }}
                        >
                          delete
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {error && <div className="text-xs text-red-700 border border-red-200 bg-red-50 p-2 rounded">{error}</div>}
            {loading && <div className="text-xs text-zinc-500 font-mono">Working...</div>}
          </>
        )}
      </aside>

      <main className="flex-1 p-6 overflow-auto">
        {tab === "classes" && (
          <section className="max-w-3xl space-y-4">
            <h2 className="text-xl font-semibold">Label Classes</h2>
            <p className="text-sm text-zinc-600">Class names are persistent and must have no spaces.</p>
            <div className="flex gap-2">
              <input
                value={newClass}
                onChange={(e) => setNewClass(e.target.value)}
                placeholder="e.g. gull, tern, heron"
                className="border border-zinc-300 rounded px-3 py-2 w-96"
              />
              <button onClick={createClass} className="px-4 py-2 border border-zinc-300 rounded bg-zinc-900 text-white">
                Add
              </button>
            </div>
            <div className="border rounded bg-white">
              {classes.map((c) => (
                <div key={c.id} className="flex items-center justify-between p-3 border-b last:border-b-0">
                  <span className="font-mono text-sm">{c.name}</span>
                  <button className="text-xs text-red-700" onClick={() => removeClass(c.id)}>
                    delete
                  </button>
                </div>
              ))}
              {classes.length === 0 && <p className="p-3 text-sm text-zinc-500">No classes yet.</p>}
            </div>
          </section>
        )}

        {tab === "projects" && (
          <section className="space-y-5">
            <h2 className="text-xl font-semibold">Projects</h2>
            <div className="flex gap-2">
              <input
                value={newProject}
                onChange={(e) => setNewProject(e.target.value)}
                placeholder="Project name"
                className="border border-zinc-300 rounded px-3 py-2 w-96"
              />
              <button onClick={createProject} className="px-4 py-2 border border-zinc-300 rounded bg-zinc-900 text-white">
                Create
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="border rounded bg-white">
                {projects.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setSelectedProjectId(p.id)}
                    className={`w-full text-left p-3 border-b last:border-b-0 ${
                      selectedProjectId === p.id ? "bg-zinc-100" : ""
                    }`}
                  >
                    <div className="font-medium">{p.name}</div>
                    <div className="text-xs text-zinc-500">{p.tile_count} tiles</div>
                  </button>
                ))}
                {projects.length === 0 && <p className="p-3 text-sm text-zinc-500">No projects yet.</p>}
              </div>

              <div className="col-span-2 border rounded bg-white p-4 space-y-4">
                {!selectedProject && <p className="text-sm text-zinc-500">Select a project.</p>}
                {selectedProject && (
                  <>
                    <div>
                      <h3 className="font-semibold">{selectedProject.name}</h3>
                      <p className="text-xs text-zinc-500">Folder: /static/{selectedProject.slug}</p>
                    </div>

                    <div className="border rounded p-3 bg-zinc-50">
                      <p className="text-sm mb-2">Ingest orthomosaic (GeoTIFF)</p>
                      <input
                        type="file"
                        accept=".tif,.tiff,.jpg,.jpeg,.png"
                        onChange={(e) => ingestOrthomosaic(e.target.files?.[0] || null)}
                      />
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={() => setTab("labeler")}
                        disabled={projectTiles.length === 0}
                        className="px-4 py-2 border border-zinc-300 rounded"
                      >
                        Open Labeler
                      </button>
                      <button onClick={exportProject} className="px-4 py-2 border border-zinc-300 rounded">
                        Export YOLO
                      </button>
                    </div>

                    <div className="text-sm text-zinc-600">
                      {projectMosaics.length} orthomosaic(s), {projectTiles.length} tile(s)
                    </div>
                  </>
                )}
              </div>
            </div>
          </section>
        )}

        {tab === "labeler" && (
          <section className="h-[calc(100vh-3rem)] space-y-3 flex flex-col">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Labeling</h2>
              <div className="text-sm text-zinc-600">D: one-shot draw | ← →: previous/next tile</div>
            </div>

            {!selectedProject && <p>Select a project first.</p>}
            {selectedProject && projectTiles.length === 0 && <p>No tiles in this project yet.</p>}

            {selectedProject && projectTiles.length > 0 && selectedTile && (
              <div className="flex-1 border rounded bg-zinc-900 p-2 flex items-center justify-center">
                <div
                  ref={stageRef}
                  className={`relative select-none ${drawMode ? "cursor-crosshair" : "cursor-default"}`}
                  style={{ width: "min(96vw, 1500px)", height: "min(88vh, 1500px)" }}
                  onMouseDown={onStageDown}
                  onMouseMove={onStageMove}
                  onMouseUp={onStageUp}
                  onMouseLeave={onStageUp}
                >
                  <img
                    src={`${API_ORIGIN}${selectedTile.rel_path}`}
                    alt={selectedTile.file_name}
                    className="absolute inset-0 h-full w-full object-contain"
                    draggable={false}
                  />

                  <svg className="absolute inset-0 h-full w-full">
                    {stageRef.current &&
                      displayedAnnotations.map((ann) => {
                        const box = stageRef.current.getBoundingClientRect()
                        const r = fromNormRect(ann, box.width, box.height)
                        const color = colorForClassId(ann.class_id)
                        const isSelected = selectedAnnotationId === ann.id

                        return (
                          <g key={ann.id}>
                            <rect
                              x={r.x}
                              y={r.y}
                              width={r.w}
                              height={r.h}
                              fill={hexToRgba(color, isSelected ? 0.25 : 0.15)}
                              stroke={color}
                              strokeWidth={isSelected ? "3" : "2"}
                            />
                            <text x={r.x + 6} y={Math.max(14, r.y + 14)} fill={color} fontSize="12">
                              {ann.class_name}
                            </text>

                            {isSelected && !drawMode && (
                              <>
                                <rect
                                  x={r.x + r.w - 6}
                                  y={r.y + r.h - 6}
                                  width={12}
                                  height={12}
                                  fill={color}
                                  stroke="white"
                                  strokeWidth="1"
                                />
                                <g
                                  onMouseDown={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    deleteAnnotation(ann.id)
                                  }}
                                  style={{ cursor: "pointer" }}
                                >
                                  <rect
                                    x={r.x + r.w - 18}
                                    y={Math.max(0, r.y - 18)}
                                    width={18}
                                    height={18}
                                    rx={4}
                                    fill="rgba(127, 29, 29, 0.9)"
                                  />
                                  <text x={r.x + r.w - 12} y={Math.max(0, r.y - 6)} fontSize="12" fill="white">
                                    x
                                  </text>
                                </g>
                              </>
                            )}
                          </g>
                        )
                      })}

                    {dragRect && (
                      <rect
                        x={Math.min(dragRect.x1, dragRect.x2)}
                        y={Math.min(dragRect.y1, dragRect.y2)}
                        width={Math.abs(dragRect.x2 - dragRect.x1)}
                        height={Math.abs(dragRect.y2 - dragRect.y1)}
                        fill="rgba(255, 255, 255, 0.2)"
                        stroke="white"
                        strokeWidth="2"
                      />
                    )}
                  </svg>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}
