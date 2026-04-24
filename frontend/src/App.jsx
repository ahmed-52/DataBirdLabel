import { useEffect, useMemo, useRef, useState } from "react"
import { Routes, Route, NavLink, useNavigate, useParams } from "react-router-dom"
import { db } from "./firebase"
import {
  collection, doc, getDoc, getDocs, setDoc, updateDoc,
  query, where, orderBy, limit, Timestamp, runTransaction,
} from "firebase/firestore"

const COLORS = [
  "#22c55e", "#f97316", "#06b6d4", "#eab308", "#ef4444",
  "#3b82f6", "#84cc16", "#ec4899", "#14b8a6", "#a855f7",
  "#10b981", "#f43f5e",
]

const GATE_KEYWORD = import.meta.env.VITE_GATE_KEYWORD || ""
const SESSION_TIMEOUT_MS = 2 * 60 * 1000 // 2 minutes
const HEARTBEAT_MS = 30000

function colorFor(id) {
  return COLORS[(((Number(id) || 0) * 2654435761) >>> 0) % COLORS.length]
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
function hexRgba(hex, a) {
  const n = Number.parseInt(hex.replace("#", ""), 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`
}
function toNorm(rect, cw, ch) {
  const x1 = Math.min(rect.x1, rect.x2), y1 = Math.min(rect.y1, rect.y2)
  const x2 = Math.max(rect.x1, rect.x2), y2 = Math.max(rect.y1, rect.y2)
  const w = x2 - x1, h = y2 - y1
  if (w < 4 || h < 4) return null
  return { x_center: (x1 + w / 2) / cw, y_center: (y1 + h / 2) / ch, width: w / cw, height: h / ch }
}
function fromNorm(a, cw, ch) {
  const w = a.width * cw, h = a.height * ch
  return { x: a.x_center * cw - w / 2, y: a.y_center * ch - h / 2, w, h }
}
function normPx(px, cw, ch) {
  const x = clamp(px.x, 0, cw), y = clamp(px.y, 0, ch)
  const w = clamp(px.w, 0, cw - x), h = clamp(px.h, 0, ch - y)
  return { x_center: (x + w / 2) / cw, y_center: (y + h / 2) / ch, width: w / cw, height: h / ch }
}
function Kbd({ children }) {
  return <kbd className="inline-flex items-center justify-center h-7 min-w-[28px] px-1.5 rounded-md bg-zinc-100 border border-zinc-300 text-xs font-bold font-mono text-zinc-700 shadow-[0_1px_0_1px_rgba(0,0,0,0.08)]">{children}</kbd>
}

// ─── Firestore helpers ───

async function fsGetAll(col, ...constraints) {
  const q = query(collection(db, col), ...constraints)
  const snap = await getDocs(q)
  return snap.docs.map(d => d.data())
}

async function fsGet(col, id) {
  const snap = await getDoc(doc(db, col, String(id)))
  return snap.exists() ? snap.data() : null
}

async function nextId(counterName) {
  const counterRef = doc(db, "_counters", counterName)
  return runTransaction(db, async (txn) => {
    const snap = await txn.get(counterRef)
    const current = snap.exists() ? snap.data().value : 0
    const next = current + 1
    txn.set(counterRef, { value: next })
    return next
  })
}

// ─── Session Lock ───

function getSessionId() { return sessionStorage.getItem("dbl_session") }
function setSessionStorage(id) { sessionStorage.setItem("dbl_session", id) }
function clearSession() { sessionStorage.removeItem("dbl_session") } // labeler persists across sessions
function getLabeler() { return localStorage.getItem("dbl_labeler") || "" }
function setLabeler(name) { localStorage.setItem("dbl_labeler", name) }

async function validateSession(sid) {
  const snap = await getDoc(doc(db, "sessions", "_lock"))
  if (!snap.exists()) return false
  return snap.data().session_id === sid
}

async function claimSession() {
  const sid = crypto.randomUUID()
  const lockRef = doc(db, "sessions", "_lock")
  await runTransaction(db, async (txn) => {
    const snap = await txn.get(lockRef)
    if (snap.exists()) {
      const data = snap.data()
      const cutoff = new Date(Date.now() - SESSION_TIMEOUT_MS)
      if (data.last_heartbeat.toDate() > cutoff) {
        throw new Error("Another labeler is active. Try again later.")
      }
    }
    txn.set(lockRef, {
      session_id: sid,
      created_at: Timestamp.now(),
      last_heartbeat: Timestamp.now(),
    })
  })
  return sid
}

// Returns true if heartbeat succeeded, false if session lost
let heartbeatFailures = 0
async function heartbeat(sid) {
  try {
    const lockRef = doc(db, "sessions", "_lock")
    const snap = await getDoc(lockRef)
    if (snap.exists() && snap.data().session_id === sid) {
      await updateDoc(lockRef, { last_heartbeat: Timestamp.now() })
      heartbeatFailures = 0
      return true
    } else {
      // Lock stolen or deleted
      return false
    }
  } catch {
    heartbeatFailures++
    // Only kick after 3 consecutive failures (1.5 min of no connection)
    return heartbeatFailures < 3
  }
}

async function releaseSession(sid) {
  try {
    const lockRef = doc(db, "sessions", "_lock")
    await runTransaction(db, async (txn) => {
      const snap = await txn.get(lockRef)
      if (snap.exists() && snap.data().session_id === sid) {
        txn.delete(lockRef)
      }
    })
  } catch {}
}

// ─── Gate Screen ───

function GatePage({ onSuccess }) {
  const [keyword, setKeyword] = useState("")
  const [labeler, setLabelerLocal] = useState(() => getLabeler())
  const [error, setError] = useState("")
  const [checking, setChecking] = useState(false)

  async function submit() {
    const name = labeler.trim()
    if (!name) { setError("Enter your name."); return }
    if (!keyword.trim()) { setError("Enter the keyword."); return }
    setChecking(true); setError("")
    try {
      if (keyword.trim() !== GATE_KEYWORD) throw new Error("Wrong keyword.")
      const sid = await claimSession()
      setSessionStorage(sid)
      setLabeler(name)
      onSuccess(sid)
    } catch (e) { setError(e.message) }
    finally { setChecking(false) }
  }

  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <img src="/databird.png" alt="" className="h-12 w-12 object-contain mx-auto mb-4 opacity-80" />
          <h1 className="text-xl font-bold text-white tracking-tight font-display">DataBirdLabel</h1>
          <p className="text-sm text-zinc-500 mt-1">Enter your name and the keyword to start labeling</p>
        </div>
        <div className="space-y-3">
          <input type="text" value={labeler}
            onChange={e => { setLabelerLocal(e.target.value); setLabeler(e.target.value) }}
            onKeyDown={e => e.key === "Enter" && submit()}
            onFocus={e => e.target.select()}
            placeholder="Your name" autoFocus
            maxLength={80}
            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 text-white text-sm focus:outline-none focus:ring-2 focus:ring-teal-600/50 focus:border-teal-600 placeholder:text-zinc-600" />
          <input type="password" value={keyword} onChange={e => setKeyword(e.target.value)}
            onKeyDown={e => e.key === "Enter" && submit()} placeholder="Keyword"
            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 text-white text-sm focus:outline-none focus:ring-2 focus:ring-teal-600/50 focus:border-teal-600 placeholder:text-zinc-600" />
          <button onClick={submit} disabled={checking}
            className="w-full py-3 rounded-lg bg-teal-700 text-white text-sm font-medium hover:bg-teal-600 transition-colors disabled:opacity-50">
            {checking ? "Checking..." : "Enter"}
          </button>
          {error && <p className="text-sm text-red-400 text-center">{error}</p>}
        </div>
      </div>
    </div>
  )
}

// ─── Shell ───

function Shell({ classes, projects, labeler, error, setError, loading, children }) {
  return (
    <div className="min-h-screen bg-background text-foreground flex">
      <aside className="w-56 border-r border-zinc-200 bg-white flex flex-col">
        <div className="p-4 border-b border-zinc-100">
          <div className="flex items-center gap-2.5">
            <img src="/databird.png" alt="" className="h-7 w-7 object-contain" />
            <div>
              <h1 className="text-sm font-bold tracking-tight font-display leading-none">DataBirdLabel</h1>
              <p className="text-[9px] uppercase tracking-widest text-zinc-400 font-mono mt-0.5">Orthomosaic Labeler</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          <NavLink to="/classes" className={({ isActive }) => `block w-full text-left px-3 py-2.5 rounded-md text-sm transition-colors ${isActive ? "bg-zinc-100 font-medium" : "text-zinc-600 hover:bg-zinc-50"}`}>Classes</NavLink>
          <NavLink to="/projects" className={({ isActive }) => `block w-full text-left px-3 py-2.5 rounded-md text-sm transition-colors ${isActive ? "bg-zinc-100 font-medium" : "text-zinc-600 hover:bg-zinc-50"}`}>Projects</NavLink>
        </nav>
        <div className="p-3 border-t border-zinc-100">
          {error && (
            <div className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-md px-3 py-2 mb-2">
              {error}<button onClick={() => setError("")} className="ml-2 text-red-400 hover:text-red-600">&times;</button>
            </div>
          )}
          {loading && <p className="text-xs text-zinc-400 font-mono animate-pulse">Working...</p>}
          {labeler && <p className="text-[11px] text-zinc-600 mt-1">Signed in as <span className="font-medium text-zinc-800">{labeler}</span></p>}
          <p className="text-[10px] text-zinc-400 mt-1">{classes.length} classes &middot; {projects.length} projects</p>
        </div>
      </aside>
      <main className="flex-1 p-8 overflow-auto">{children}</main>
    </div>
  )
}

// ─── Classes (read-only) ───

function ClassesPage({ classes }) {
  return (
    <section className="max-w-2xl">
      <div className="mb-6">
        <h2 className="text-lg font-semibold tracking-tight mb-1">Species Classes</h2>
        <p className="text-sm text-zinc-500">Classes are managed by the admin via CLI.</p>
      </div>
      {classes.length === 0 ? (
        <div className="border border-dashed border-zinc-300 rounded-lg py-12 text-center">
          <p className="text-zinc-400 text-sm">No classes defined yet</p>
          <p className="text-zinc-400 text-xs mt-1">Ask the admin to add species classes</p>
        </div>
      ) : (
        <div className="border border-zinc-200 rounded-lg overflow-hidden">
          {classes.map((c, i) => (
            <div key={c.id} className={`flex items-center gap-3 px-4 py-3 ${i < classes.length - 1 ? "border-b border-zinc-100" : ""}`}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colorFor(c.id) }} />
              <span className="font-mono text-sm">{c.name}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

// ─── Projects (read-only, with Label button) ───

function ProjectsPage({ projects, classes }) {
  const navigate = useNavigate()
  return (
    <section className="max-w-3xl">
      <div className="mb-6">
        <h2 className="text-lg font-semibold tracking-tight mb-1">Projects</h2>
        <p className="text-sm text-zinc-500">Projects are created by the admin via CLI. Select one to start labeling.</p>
      </div>
      {projects.length === 0 ? (
        <div className="border border-dashed border-zinc-300 rounded-lg py-12 text-center">
          <p className="text-zinc-400 text-sm">No projects yet</p>
          <p className="text-zinc-400 text-xs mt-1">Ask the admin to ingest an orthomosaic</p>
        </div>
      ) : (
        <div className="space-y-3">
          {projects.map(p => (
            <div key={p.id} className="border border-zinc-200 rounded-lg bg-white overflow-hidden">
              <div className="px-5 py-4 flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-sm">{p.name}</h3>
                  <p className="text-xs text-zinc-400 mt-0.5 font-mono">{p.tile_count} tiles</p>
                </div>
                {p.tile_count > 0 && (
                  <button onClick={() => navigate(`/projects/${p.id}/label`)}
                    className="px-4 py-2 rounded-md bg-teal-700 text-white text-sm font-medium hover:bg-teal-600 transition-colors">
                    Label
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

// ─── Label Page ───

function LabelPage({ classes, labeler, setError }) {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState(null)
  const [tiles, setTiles] = useState([])
  const [tileIdx, setTileIdx] = useState(0)
  const [anns, setAnns] = useState([])
  const [comments, setComments] = useState([])
  const [commentDraft, setCommentDraft] = useState("")
  const [postingComment, setPostingComment] = useState(false)
  const [classId, setClassId] = useState(classes[0]?.id || null)
  const [selAnn, setSelAnn] = useState(null)
  const [draw, setDraw] = useState(false)
  const [drag, setDrag] = useState(null)
  const [interaction, setInteraction] = useState(null)
  const [editing, setEditing] = useState(null)
  const [zoom, setZoom] = useState(1)
  const ref = useRef(null)

  const pid = Number(projectId)
  const tile = tiles[tileIdx] || null
  const tileW = tile?.width || 1280
  const tileH = tile?.height || 1280
  const sw = Math.round(tileW * zoom)
  const sh = Math.round(tileH * zoom)

  const displayed = useMemo(() => {
    return anns.map(a => editing && editing.id === a.id ? { ...a, ...editing } : a)
  }, [anns, editing])

  // Build class lookup once
  const classMap = useMemo(() => {
    const m = {}
    for (const c of classes) m[c.id] = c.name
    return m
  }, [classes])

  async function load() {
    try {
      const p = await fsGet("projects", pid)
      if (!p) { setError("Project not found"); return }
      setProject(p)
      let t
      try {
        t = await fsGetAll("tiles", where("project_id", "==", pid), orderBy("id"))
      } catch {
        t = await fsGetAll("tiles", where("project_id", "==", pid))
        t.sort((a, b) => a.id - b.id)
      }
      setTiles(t)
      setTileIdx(0)

      // Preload first few tile images
      for (let i = 0; i < Math.min(5, t.length); i++) {
        const img = new Image()
        img.src = t[i].storage_url
      }
    } catch (e) {
      console.error("[load]", e)
      setError(e.message)
    }
  }

  // Track which tile has pending writes
  const pendingTileRef = useRef(null)

  async function loadAnns(tid) {
    if (!tid) { setAnns([]); return }
    // Only skip if pending writes are for THIS tile
    if (pendingRef.current > 0 && pendingTileRef.current === tid) return
    let rows
    try {
      rows = await fsGetAll("annotations", where("tile_id", "==", tid), orderBy("id"))
    } catch {
      rows = await fsGetAll("annotations", where("tile_id", "==", tid))
      rows.sort((a, b) => a.id - b.id)
    }
    if (pendingRef.current > 0 && pendingTileRef.current === tid) return
    // Soft-deleted annotations stay in Firestore for audit but are hidden from the UI.
    // Legacy rows (pre-audit schema) have no `deleted` field → treated as not deleted.
    rows = rows.filter(a => a.deleted !== true)
    for (const a of rows) {
      a.class_name = classMap[a.class_id] || `class_${a.class_id}`
      a.secondary_class_name = a.secondary_class_id ? (classMap[a.secondary_class_id] || `class_${a.secondary_class_id}`) : null
    }
    setAnns(rows)
  }

  async function loadComments(tid) {
    if (!tid) { setComments([]); return }
    let rows = await fsGetAll("comments", where("tile_id", "==", tid))
    rows.sort((a, b) => a.id - b.id)
    setComments(rows)
  }

  async function submitComment() {
    const text = commentDraft.trim()
    if (!text || !tile || postingComment) return
    setPostingComment(true)
    const originTileId = tile.id
    try {
      const cid = await nextId("comments")
      const data = {
        id: cid,
        tile_id: originTileId,
        author: labeler || "anonymous",
        text,
        created_at: new Date().toISOString(),
      }
      await setDoc(doc(db, "comments", String(cid)), data)
      // Only append to UI state if we're still on the same tile
      if (tile?.id === originTileId) {
        setComments(prev => [...prev, data])
      }
      setCommentDraft("")
    } catch (e) {
      setError(e.message)
    } finally {
      setPostingComment(false)
    }
  }

  // Preload adjacent tiles when tile index changes
  useEffect(() => {
    for (let offset = 1; offset <= 3; offset++) {
      const next = tiles[tileIdx + offset]
      const prev = tiles[tileIdx - offset]
      if (next) { const img = new Image(); img.src = next.storage_url }
      if (prev) { const img = new Image(); img.src = prev.storage_url }
    }
  }, [tileIdx, tiles])

  useEffect(() => { load().catch(e => setError(e.message)) }, [pid])
  useEffect(() => {
    // Reset pending state on tile change — pending writes for old tile will still complete in Firestore
    // but won't corrupt current tile's annotation array
    pendingTileRef.current = tile?.id || null
    if (tile) {
      loadAnns(tile.id).catch(e => setError(e.message))
      loadComments(tile.id).catch(e => setError(e.message))
    } else {
      setAnns([])
      setComments([])
    }
    setSelAnn(null)
    setEditing(null)
    setInteraction(null)
    setCommentDraft("")
  }, [tile?.id])
  useEffect(() => { if (!classId && classes.length) setClassId(classes[0].id) }, [classes])

  useEffect(() => {
    function onKey(e) {
      const tag = (e.target?.tagName || "").toLowerCase()
      if (tag === "input" || tag === "textarea" || tag === "select") return
      if (e.key.toLowerCase() === "d") setDraw(d => !d)
      if (e.key === "ArrowRight") setTileIdx(i => Math.min(i + 1, Math.max(0, tiles.length - 1)))
      if (e.key === "ArrowLeft") setTileIdx(i => Math.max(i - 1, 0))
      if (e.key === "+" || e.key === "=") { e.preventDefault(); setZoom(z => clamp(z + 0.1, 0.8, 1)) }
      if (e.key === "-" || e.key === "_") { e.preventDefault(); setZoom(z => clamp(z - 0.1, 0.8, 1)) }
      if (e.key === "0") { e.preventDefault(); setZoom(1) }
      if (!draw && selAnn && (e.key === "Delete" || e.key === "Backspace")) { e.preventDefault(); delAnn(selAnn) }
      // 1-4 → switch active class to classes[0..3]. Ignored if modifier keys are held
      // or the class slot doesn't exist.
      if (["1", "2", "3", "4"].includes(e.key) && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const c = classes[Number(e.key) - 1]
        if (c) { e.preventDefault(); setClassId(c.id) }
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [tiles.length, draw, selAnn, classes])

  function pt(e) {
    if (!ref.current) return null
    const r = ref.current.getBoundingClientRect()
    // Convert DOM pixels to tile-pixel space (zoom-independent)
    const scaleX = tileW / r.width
    const scaleY = tileH / r.height
    const x = (e.clientX - r.left) * scaleX
    const y = (e.clientY - r.top) * scaleY
    return { x: clamp(x, 0, tileW), y: clamp(y, 0, tileH), w: tileW, h: tileH }
  }

  function hitTest(x, y, sw, sh) {
    for (let i = displayed.length - 1; i >= 0; i--) {
      const r = fromNorm(displayed[i], sw, sh)
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return { ann: displayed[i], rect: r }
    }
    return null
  }

  function hitHandle(p, rect) {
    const hs = 7 / zoom // consistent hit area regardless of zoom
    const { x, y, w, h } = rect
    const handles = [
      { name: "nw", cx: x, cy: y, cursor: "nwse-resize" },
      { name: "n",  cx: x + w / 2, cy: y, cursor: "ns-resize" },
      { name: "ne", cx: x + w, cy: y, cursor: "nesw-resize" },
      { name: "e",  cx: x + w, cy: y + h / 2, cursor: "ew-resize" },
      { name: "se", cx: x + w, cy: y + h, cursor: "nwse-resize" },
      { name: "s",  cx: x + w / 2, cy: y + h, cursor: "ns-resize" },
      { name: "sw", cx: x, cy: y + h, cursor: "nesw-resize" },
      { name: "w",  cx: x, cy: y + h / 2, cursor: "ew-resize" },
    ]
    for (const handle of handles) {
      if (Math.abs(p.x - handle.cx) <= hs && Math.abs(p.y - handle.cy) <= hs) return handle
    }
    return null
  }

  // Pending writes — prevents loadAnns from overwriting optimistic state
  const pendingRef = useRef(0)
  // Local ID counter for instant annotation creation
  const localIdRef = useRef(-1)
  // Track pending creates: tempId -> { cancelled, latestUpd }
  const pendingCreates = useRef(new Map())

  function submitRect(rect) {
    if (!tile || !classId) return
    const n = toNorm(rect, tileW, tileH)
    if (!n) return

    const originTileId = tile.id // capture which tile this annotation belongs to
    const tempId = localIdRef.current--
    const className = classMap[classId] || `class_${classId}`
    const nowIso = new Date().toISOString()
    const tempData = {
      id: tempId, tile_id: originTileId, class_id: classId, class_name: className, ...n,
      secondary_class_id: null,
      created_at: nowIso, created_by: labeler || null,
      updated_at: nowIso, updated_by: labeler || null,
      deleted: false,
    }
    setAnns(prev => [...prev, tempData])

    const pending = { cancelled: false, latestUpd: null }
    pendingCreates.current.set(tempId, pending)

    pendingRef.current++
    pendingTileRef.current = originTileId
    ;(async () => {
      try {
        if (pending.cancelled) return
        const aid = await nextId("annotations")
        if (pending.cancelled) return

        let finalData = { ...tempData, id: aid }
        if (pending.latestUpd) finalData = { ...finalData, ...pending.latestUpd }
        const { class_name: _, ...firestoreData } = finalData
        await setDoc(doc(db, "annotations", String(aid)), firestoreData)

        // Only update UI if we're still on the same tile
        if (!pending.cancelled && pendingTileRef.current === originTileId) {
          setAnns(prev => prev.map(a => a.id === tempId ? { ...finalData, class_name: className } : a))
        }
      } catch (e) {
        if (!pending.cancelled && pendingTileRef.current === originTileId) {
          setAnns(prev => prev.filter(a => a.id !== tempId))
        }
        setError(e.message)
      } finally {
        pendingCreates.current.delete(tempId)
        pendingRef.current--
      }
    })()
  }

  function updateAnnNow(id, upd) {
    // Stamp who/when on every edit so we always know the last person to touch a box.
    const stamped = { ...upd, updated_at: new Date().toISOString(), updated_by: labeler || null }
    // Instant local update (also refresh denormalized class names when ids change)
    setAnns(prev => prev.map(a => a.id === id ? {
      ...a, ...stamped,
      class_name: stamped.class_id ? (classMap[stamped.class_id] || a.class_name) : a.class_name,
      secondary_class_name: "secondary_class_id" in stamped
        ? (stamped.secondary_class_id ? (classMap[stamped.secondary_class_id] || `class_${stamped.secondary_class_id}`) : null)
        : a.secondary_class_name,
    } : a))

    // Queue update for pending temp-ID annotations
    if (id < 0) {
      const pending = pendingCreates.current.get(id)
      if (pending) pending.latestUpd = { ...(pending.latestUpd || {}), ...stamped }
      return
    }
    pendingRef.current++
    updateDoc(doc(db, "annotations", String(id)), stamped)
      .catch(e => setError(e.message))
      .finally(() => { pendingRef.current-- })
  }

  function delAnn(id) {
    setAnns(prev => prev.filter(a => a.id !== id))
    if (selAnn === id) setSelAnn(null)

    // Temp-ID (never persisted): just cancel the pending create, nothing to soft-delete.
    if (id < 0) {
      const pending = pendingCreates.current.get(id)
      if (pending) pending.cancelled = true
      return
    }
    // Persisted annotation: soft-delete so we keep the audit trail (who created, who deleted, when).
    const nowIso = new Date().toISOString()
    pendingRef.current++
    updateDoc(doc(db, "annotations", String(id)), {
      deleted: true,
      deleted_at: nowIso,
      deleted_by: labeler || null,
      updated_at: nowIso,
      updated_by: labeler || null,
    })
      .catch(e => setError(e.message))
      .finally(() => { pendingRef.current-- })
  }

  function onDown(e) {
    const p = pt(e); if (!p) return
    if (draw) { setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y }); return }
    if (selAnn) {
      const selData = displayed.find(a => a.id === selAnn)
      if (selData) {
        const r = fromNorm(selData, p.w, p.h)
        const handle = hitHandle(p, r)
        if (handle) { setInteraction({ type: "handle", id: selAnn, handle: handle.name, rect: r }); return }
      }
    }
    const hit = hitTest(p.x, p.y, p.w, p.h)
    if (!hit) { setSelAnn(null); return }
    setSelAnn(hit.ann.id); setClassId(hit.ann.class_id)
    setInteraction({ type: "move", id: hit.ann.id, ox: p.x - hit.rect.x, oy: p.y - hit.rect.y, rect: hit.rect })
  }

  function onMove(e) {
    const p = pt(e); if (!p) return
    if (draw && drag) { setDrag(prev => prev ? { ...prev, x2: p.x, y2: p.y } : prev); return }
    if (!interaction) {
      if (selAnn && ref.current) {
        const selData = displayed.find(a => a.id === selAnn)
        if (selData) {
          const r = fromNorm(selData, p.w, p.h)
          const handle = hitHandle(p, r)
          ref.current.style.cursor = handle ? handle.cursor : (draw ? "crosshair" : "default")
        }
      }
      return
    }
    if (interaction.type === "move") {
      const { w, h } = interaction.rect
      const x = clamp(p.x - interaction.ox, 0, p.w - w), y = clamp(p.y - interaction.oy, 0, p.h - h)
      setEditing({ id: interaction.id, ...normPx({ x, y, w, h }, p.w, p.h) })
    }
    if (interaction.type === "handle") {
      const { rect, handle } = interaction
      const MIN = 6
      let x = rect.x, y = rect.y, w = rect.w, h = rect.h
      if (handle === "nw") { w = clamp(rect.x + rect.w - p.x, MIN, rect.x + rect.w); h = clamp(rect.y + rect.h - p.y, MIN, rect.y + rect.h); x = rect.x + rect.w - w; y = rect.y + rect.h - h }
      if (handle === "ne") { w = clamp(p.x - rect.x, MIN, p.w - rect.x); h = clamp(rect.y + rect.h - p.y, MIN, rect.y + rect.h); y = rect.y + rect.h - h }
      if (handle === "se") { w = clamp(p.x - rect.x, MIN, p.w - rect.x); h = clamp(p.y - rect.y, MIN, p.h - rect.y) }
      if (handle === "sw") { w = clamp(rect.x + rect.w - p.x, MIN, rect.x + rect.w); h = clamp(p.y - rect.y, MIN, p.h - rect.y); x = rect.x + rect.w - w }
      if (handle === "n") { h = clamp(rect.y + rect.h - p.y, MIN, rect.y + rect.h); y = rect.y + rect.h - h }
      if (handle === "s") { h = clamp(p.y - rect.y, MIN, p.h - rect.y) }
      if (handle === "w") { w = clamp(rect.x + rect.w - p.x, MIN, rect.x + rect.w); x = rect.x + rect.w - w }
      if (handle === "e") { w = clamp(p.x - rect.x, MIN, p.w - rect.x) }
      setEditing({ id: interaction.id, ...normPx({ x, y, w, h }, p.w, p.h) })
    }
  }

  function onUp() {
    if (draw && drag) { const d = drag; setDrag(null); setDraw(false); submitRect(d); return }
    if (!interaction) return
    const upd = editing ? { x_center: editing.x_center, y_center: editing.y_center, width: editing.width, height: editing.height } : null
    const id = interaction.id
    // Update anns FIRST so the position persists, THEN clear editing
    if (upd) updateAnnNow(id, upd)
    setInteraction(null)
    setEditing(null)
  }

  if (!project) return null

  const activeClass = classes.find(c => c.id === classId)

  return (
    <div className="h-screen flex flex-col">
      <div className="border-b border-zinc-200 bg-white px-5 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={() => navigate("/projects")} className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors flex items-center gap-1">
            <span className="text-lg leading-none">&larr;</span> Projects
          </button>
          <div className="h-5 w-px bg-zinc-200" />
          <h2 className="text-sm font-semibold">{project.name}</h2>
          <span className="text-xs text-zinc-400 font-mono">Tile {tiles.length ? tileIdx + 1 : 0} / {tiles.length}</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5"><Kbd>D</Kbd><span className="text-xs text-zinc-500 font-medium">Draw</span></div>
          <div className="h-4 w-px bg-zinc-200" />
          <div className="flex items-center gap-1.5"><Kbd>1</Kbd><Kbd>-</Kbd><Kbd>4</Kbd><span className="text-xs text-zinc-500 font-medium">Class</span></div>
          <div className="h-4 w-px bg-zinc-200" />
          <div className="flex items-center gap-1.5"><Kbd>+</Kbd><Kbd>-</Kbd><span className="text-xs text-zinc-500 font-medium">Zoom</span></div>
          <div className="h-4 w-px bg-zinc-200" />
          <div className="flex items-center gap-1.5"><Kbd>0</Kbd><span className="text-xs text-zinc-500 font-medium">Reset</span></div>
          <div className="h-4 w-px bg-zinc-200" />
          <div className="flex items-center gap-1.5"><Kbd>&larr;</Kbd><Kbd>&rarr;</Kbd><span className="text-xs text-zinc-500 font-medium">Tiles</span></div>
          {labeler && (<>
            <div className="h-4 w-px bg-zinc-200" />
            <span className="text-xs text-zinc-500">as <span className="font-medium text-zinc-700">{labeler}</span></span>
          </>)}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-64 border-r border-zinc-200 bg-white p-4 flex flex-col gap-4 overflow-y-auto shrink-0">
          {activeClass && (
            <div className="rounded-lg px-3 py-2.5 flex items-center gap-2.5" style={{ backgroundColor: hexRgba(colorFor(activeClass.id), 0.12), border: `1.5px solid ${colorFor(activeClass.id)}` }}>
              <span className="h-3.5 w-3.5 rounded-full shrink-0" style={{ backgroundColor: colorFor(activeClass.id) }} />
              <div>
                <p className="text-[10px] uppercase tracking-wider font-medium" style={{ color: colorFor(activeClass.id) }}>Drawing as</p>
                <p className="text-sm font-semibold" style={{ color: colorFor(activeClass.id) }}>{activeClass.name}</p>
              </div>
            </div>
          )}
          <div>
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2 block">Classes</label>
            <div className="space-y-0.5">
              {classes.map((c, i) => (
                <button key={c.id} onClick={() => setClassId(c.id)}
                  className={`w-full text-left text-sm px-3 py-2 rounded-md flex items-center gap-2.5 transition-colors ${classId === c.id ? "font-medium" : "hover:bg-zinc-50 text-zinc-600"}`}
                  style={classId === c.id ? { backgroundColor: hexRgba(colorFor(c.id), 0.1) } : {}}>
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: colorFor(c.id) }} />
                  <span className="flex-1 truncate">{c.name}</span>
                  {i < 4 && <span className="text-[10px] font-mono text-zinc-400 shrink-0">{i + 1}</span>}
                </button>
              ))}
            </div>
          </div>
          <button onClick={() => setDraw(d => !d)}
            className={`w-full px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${draw ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200"}`}>
            {draw ? "Drawing..." : "Draw Box"}
          </button>
          <div className="flex gap-1">
            <button className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs hover:bg-zinc-50" onClick={() => setZoom(z => clamp(z - 0.1, 0.8, 1))}>-</button>
            <button className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs hover:bg-zinc-50" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
            <button className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs hover:bg-zinc-50" onClick={() => setZoom(z => clamp(z + 0.1, 0.8, 1))}>+</button>
          </div>
          <div className="flex gap-1">
            <button className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs hover:bg-zinc-50" onClick={() => setTileIdx(i => Math.max(i - 1, 0))}>&larr; Prev</button>
            <button className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs hover:bg-zinc-50" onClick={() => setTileIdx(i => Math.min(i + 1, tiles.length - 1))}>Next &rarr;</button>
          </div>
          <div>
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2 block">Annotations ({anns.length})</label>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {anns.map((a, idx) => (
                <div key={a.id} onClick={() => setSelAnn(a.id)}
                  className={`text-xs px-2.5 py-1.5 rounded-md flex items-center justify-between gap-2 cursor-pointer transition-colors ${selAnn === a.id ? "bg-zinc-100 ring-1 ring-zinc-300" : "hover:bg-zinc-50"}`}>
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <span className="font-mono text-[10px] text-zinc-500 shrink-0 w-5 text-right">#{idx + 1}</span>
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: colorFor(a.class_id) }} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate leading-tight">
                        {a.class_name}
                        {a.secondary_class_id && <span className="text-zinc-400"> / ?{a.secondary_class_name || `class_${a.secondary_class_id}`}</span>}
                      </div>
                      {a.created_by && <div className="text-[10px] text-zinc-400 truncate leading-tight">{a.created_by}</div>}
                    </div>
                  </div>
                  <button onClick={e => { e.stopPropagation(); delAnn(a.id) }} className="text-zinc-400 hover:text-red-600 transition-colors shrink-0">&times;</button>
                </div>
              ))}
              {anns.length === 0 && <p className="text-xs text-zinc-400 py-2">No annotations on this tile</p>}
            </div>
          </div>
          {selAnn && (() => {
            const selData = anns.find(a => a.id === selAnn)
            const currentSecondary = selData?.secondary_class_id || ""
            return (
              <div className="space-y-2">
                <button onClick={() => {
                    if (!selAnn || !classId) return
                    // If the new primary matches the existing secondary, drop the secondary so we never store primary==secondary.
                    const upd = { class_id: classId }
                    if (selData?.secondary_class_id === classId) upd.secondary_class_id = null
                    updateAnnNow(selAnn, upd)
                  }}
                  disabled={!selAnn || !classId || selData?.class_id === classId}
                  className="w-full px-3 py-2 rounded-md border border-zinc-200 text-xs hover:bg-zinc-50 disabled:opacity-40 transition-colors">
                  Reassign to active class
                </button>
                <div>
                  <label className="text-[10px] uppercase tracking-wider font-medium text-zinc-500 mb-1 block">Possible 2nd species (unsure)</label>
                  <select
                    value={currentSecondary}
                    onChange={e => {
                      const v = e.target.value
                      updateAnnNow(selAnn, { secondary_class_id: v ? Number(v) : null })
                    }}
                    className="w-full border border-zinc-200 rounded-md px-2 py-1.5 text-xs bg-white">
                    <option value="">None (confident)</option>
                    {classes.filter(c => c.id !== selData?.class_id).map(c => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>
              </div>
            )
          })()}

          <div className="pt-2 mt-auto border-t border-zinc-100">
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2 block">Comments ({comments.length})</label>
            <div className="space-y-2 max-h-48 overflow-y-auto bg-zinc-50 border border-zinc-200 rounded-md p-2 mb-2">
              {comments.length === 0 && <p className="text-[11px] text-zinc-400">No comments on this tile yet. Use #1, #2... to reference boxes.</p>}
              {comments.map(c => (
                <div key={c.id} className="text-[11px] leading-snug">
                  <div className="flex items-baseline gap-1.5">
                    <span className="font-medium text-zinc-700 truncate">{c.author}</span>
                    <span className="text-zinc-400 text-[10px] shrink-0">{new Date(c.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</span>
                  </div>
                  <p className="text-zinc-600 whitespace-pre-wrap break-words">{c.text}</p>
                </div>
              ))}
            </div>
            <div className="flex gap-1">
              <input type="text" value={commentDraft}
                onChange={e => setCommentDraft(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitComment() } }}
                placeholder="Add comment... (ref #1, #2)"
                maxLength={500}
                className="flex-1 border border-zinc-200 rounded-md px-2 py-1.5 text-xs bg-white focus:outline-none focus:ring-1 focus:ring-teal-600" />
              <button onClick={submitComment}
                disabled={!commentDraft.trim() || postingComment || !tile}
                className="px-3 py-1.5 rounded-md bg-teal-700 text-white text-xs font-medium hover:bg-teal-600 transition-colors disabled:opacity-40">
                {postingComment ? "…" : "Post"}
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 bg-zinc-900 flex items-center justify-center overflow-auto p-3">
          {!tile && <p className="text-zinc-500 text-sm">No tiles to label</p>}
          {tile && (
            <div ref={ref}
              className={`relative select-none ${draw ? "cursor-crosshair" : "cursor-default"}`}
              style={{ width: `${sw}px`, height: `${sh}px` }}
              onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}>
              <img src={tile.storage_url} alt={tile.file_name}
                className="absolute inset-0 h-full w-full object-contain" draggable={false} />
              <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${tileW} ${tileH}`} preserveAspectRatio="none" style={{ pointerEvents: "none" }}>
                {displayed.map((a, idx) => {
                  const r = fromNorm(a, tileW, tileH)
                  const c = colorFor(a.class_id)
                  const sel = selAnn === a.id
                  const HS = 5
                  // When a secondary (unsure) class is set, show both in the label: "primary / ?secondary"
                  // #N matches the sidebar/comments numbering so commenters can reference boxes.
                  const baseText = a.secondary_class_id
                    ? `${a.class_name} / ?${a.secondary_class_name || `class_${a.secondary_class_id}`}`
                    : a.class_name
                  const labelText = `#${idx + 1} ${baseText}`
                  const tagY = r.y > 20 ? r.y - 20 : r.y + 2
                  return (
                    <g key={a.id}>
                      <rect x={r.x} y={r.y} width={r.w} height={r.h} fill={hexRgba(c, sel ? 0.18 : 0.08)} />
                      <rect x={r.x} y={r.y} width={r.w} height={r.h} fill="none" stroke={c} strokeWidth={sel ? 2 : 1.5} strokeDasharray={sel ? "none" : "4 2"} />
                      <g>
                        <rect x={r.x} y={tagY} width={labelText.length * 7 + 12} height={18} rx={2} fill={c} opacity={sel ? 1 : 0.85} />
                        <text x={r.x + 6} y={tagY + 13} fill="white" fontSize="11" fontWeight="600" fontFamily="monospace" style={{ pointerEvents: "none" }}>{labelText}</text>
                      </g>
                      {sel && !draw && (() => {
                        const handles = [
                          { cx: r.x, cy: r.y }, { cx: r.x + r.w / 2, cy: r.y }, { cx: r.x + r.w, cy: r.y },
                          { cx: r.x + r.w, cy: r.y + r.h / 2 }, { cx: r.x + r.w, cy: r.y + r.h },
                          { cx: r.x + r.w / 2, cy: r.y + r.h }, { cx: r.x, cy: r.y + r.h }, { cx: r.x, cy: r.y + r.h / 2 },
                        ]
                        return handles.map((h, i) => (
                          <rect key={i} x={h.cx - HS} y={h.cy - HS} width={HS * 2} height={HS * 2} fill="white" stroke={c} strokeWidth={1.5} rx={1} />
                        ))
                      })()}
                    </g>
                  )
                })}
                {drag && (() => {
                  const dx = Math.min(drag.x1, drag.x2), dy = Math.min(drag.y1, drag.y2)
                  const dw = Math.abs(drag.x2 - drag.x1), dh = Math.abs(drag.y2 - drag.y1)
                  const dc = activeClass ? colorFor(activeClass.id) : "#fff"
                  return (
                    <g>
                      <rect x={dx} y={dy} width={dw} height={dh} fill={hexRgba(dc, 0.12)} stroke={dc} strokeWidth={2} />
                      <line x1={dx} y1={dy} x2={dx + dw} y2={dy} stroke={dc} strokeWidth={0.5} strokeDasharray="3 3" />
                      <line x1={dx} y1={dy + dh} x2={dx + dw} y2={dy + dh} stroke={dc} strokeWidth={0.5} strokeDasharray="3 3" />
                      <line x1={dx} y1={dy} x2={dx} y2={dy + dh} stroke={dc} strokeWidth={0.5} strokeDasharray="3 3" />
                      <line x1={dx + dw} y1={dy} x2={dx + dw} y2={dy + dh} stroke={dc} strokeWidth={0.5} strokeDasharray="3 3" />
                      {dw > 30 && dh > 15 && (
                        <text x={dx + dw / 2} y={dy + dh / 2 + 4} textAnchor="middle" fill="white" fontSize="10" fontFamily="monospace" opacity={0.7}>{Math.round(dw)}&times;{Math.round(dh)}</text>
                      )}
                      {activeClass && (
                        <g>
                          <rect x={dx} y={dy > 20 ? dy - 20 : dy + 2} width={activeClass.name.length * 7 + 12} height={18} rx={2} fill={dc} />
                          <text x={dx + 6} y={(dy > 20 ? dy - 20 : dy + 2) + 13} fill="white" fontSize="11" fontWeight="600" fontFamily="monospace">{activeClass.name}</text>
                        </g>
                      )}
                    </g>
                  )
                })()}
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── App ───

export default function App() {
  const [session, setSession] = useState(getSessionId())
  const [labeler, setLabelerState] = useState(getLabeler())
  const [classes, setClasses] = useState([])
  const [projects, setProjects] = useState([])
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  async function refreshClasses() {
    setClasses(await fsGetAll("classes", orderBy("id")))
  }
  async function refreshProjects() {
    const docs = await fsGetAll("projects", orderBy("id"))
    // tile_count is stored on the project doc by the CLI ingest
    for (const p of docs) {
      if (p.tile_count === undefined) p.tile_count = 0
    }
    setProjects(docs)
  }

  // Activity tracking + heartbeat
  const lastActivity = useRef(Date.now())
  const IDLE_TIMEOUT_MS = 5 * 60 * 1000 // 5 minutes

  useEffect(() => {
    if (!session) return
    function onActivity() { lastActivity.current = Date.now() }
    window.addEventListener("mousemove", onActivity)
    window.addEventListener("mousedown", onActivity)
    window.addEventListener("keydown", onActivity)
    return () => {
      window.removeEventListener("mousemove", onActivity)
      window.removeEventListener("mousedown", onActivity)
      window.removeEventListener("keydown", onActivity)
    }
  }, [session])

  // Validate session on mount (page refresh)
  useEffect(() => {
    if (!session) return
    validateSession(session).then(valid => {
      if (!valid) { clearSession(); setSession(null) }
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!session) return
    const iv = setInterval(async () => {
      const idle = Date.now() - lastActivity.current
      if (idle > IDLE_TIMEOUT_MS) {
        // AFK — release lock and kick to gate
        releaseSession(session).catch(() => {})
        clearSession()
        setSession(null)
      } else if (idle < SESSION_TIMEOUT_MS) {
        // Only send heartbeat if user was active recently
        // This way, the lock naturally expires if they're idle > 2 min
        const ok = await heartbeat(session)
        if (!ok) {
          clearSession()
          setSession(null)
        }
      }
      // Between 2-5 min idle: don't heartbeat (let lock expire) but don't kick yet
    }, HEARTBEAT_MS)
    return () => clearInterval(iv)
  }, [session])

  // Release on tab close
  useEffect(() => {
    if (!session) return
    function onUnload() { const sid = getSessionId(); if (sid) releaseSession(sid) }
    window.addEventListener("beforeunload", onUnload)
    return () => window.removeEventListener("beforeunload", onUnload)
  }, [session])

  useEffect(() => {
    if (!session) return
    refreshClasses().catch(e => setError(e.message))
    refreshProjects().catch(e => setError(e.message))
  }, [session])

  if (!session) return <GatePage onSuccess={sid => { setSession(sid); setLabelerState(getLabeler()) }} />

  return (
    <Routes>
      <Route path="/projects/:projectId/label" element={<LabelPage classes={classes} labeler={labeler} setError={setError} />} />
      <Route path="*" element={
        <Shell classes={classes} projects={projects} labeler={labeler} error={error} setError={setError} loading={loading}>
          <Routes>
            <Route path="/classes" element={<ClassesPage classes={classes} />} />
            <Route path="/projects" element={<ProjectsPage projects={projects} classes={classes} />} />
            <Route path="*" element={<ClassesPage classes={classes} />} />
          </Routes>
        </Shell>
      } />
    </Routes>
  )
}
