import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  BarChart3,
  Bell,
  Bot,
  Camera,
  CheckCircle2,
  Download,
  FileText,
  RefreshCw,
  Settings,
  ShieldAlert,
  SlidersHorizontal,
  UserCheck,
  Users,
  XCircle,
} from 'lucide-react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin
const API_AUTH = import.meta.env.VITE_API_AUTH

type EventType =
  | 'employee_absence'
  | 'employee_presence'
  | 'visitor_shelf_dwell'
  | 'hand_to_body'
  | 'back_to_camera'
  | 'system_stream_lost'

type EventStatus = 'new' | 'confirmed' | 'dismissed'
type Severity = 'low' | 'medium' | 'high'
type Role = 'admin' | 'operator' | 'manager'
type Tab = 'overview' | 'events' | 'analytics' | 'cameras' | 'settings'

type Overview = {
  active_cameras: number
  total_cameras: number
  open_events: number
  confirmed_events: number
  absence_events: number
  suspicious_events: number
  telegram_configured: boolean
}

type Zone = {
  id: string
  name: string
  kind: 'work_area' | 'shelf' | 'entrance' | 'checkout'
}

type MonitoringCamera = {
  id: string
  name: string
  location: string
  rtsp_url: string
  status: 'online' | 'unstable' | 'offline'
  ai_status: 'running' | 'warming_up' | 'disabled'
  fps: number
  zones: Zone[]
  last_seen_at: string
  source_type?: 'demo_video' | 'rtsp' | 'mock_rtsp' | 'public_dataset' | 'public_webcam_archive'
  quality_score?: number
  uptime_minutes?: number
  last_event_title?: string | null
  last_event_at?: string | null
}

type VideoEvent = {
  id: string
  camera_id: string
  camera_name: string
  type: EventType
  severity: Severity
  title: string
  description: string
  zone: string
  detected_at: string
  snapshot_url: string
  status: EventStatus
  confidence: number
  feedback_note: string | null
  reviewed_by: string | null
  telegram_sent: boolean
  reaction_seconds?: number | null
  analysis_summary: string
  evidence_tags: string[]
}

type ShiftAnalytics = {
  report_date: string
  shift_started_at: string
  total_events: number
  open_events: number
  confirmed_events: number
  dismissed_events: number
  absence_events: number
  suspicious_events: number
  average_reaction_seconds: number
  cameras_online: number
  cameras_total: number
  telegram_configured: boolean
}

type MonitoringSettings = {
  absence_threshold_minutes: number
  shelf_dwell_seconds: number
  confidence_threshold: number
}

type TelegramButton = {
  label: string
  action: string
  callback_data: string
}

type TelegramPreview = {
  mode: 'telegram' | 'mock'
  text: string
  buttons: TelegramButton[]
}

type TelegramTestResponse = {
  configured: boolean
  sent: boolean
  mode: 'telegram' | 'mock'
  detail: string
  inline_feedback: boolean
  preview: TelegramPreview
}

type PublicVideoSource = {
  id: string
  title: string
  camera_id: string
  source_url: string
  scenario: string
  license_note: string
  supported_signals: string[]
}

type DetectionCapability = {
  id: string
  title: string
  readiness: 'demo_ready' | 'heuristic_ready' | 'pilot_needed'
  confidence: number
  what_it_checks: string
  evidence: string[]
  current_limitations: string
  tz_mapping: string
}

type Filters = {
  cameraId: string
  type: string
  status: string
  dateFrom: string
  dateTo: string
}

const EVENT_TYPE_LABELS: Record<EventType, string> = {
  employee_absence: 'Отсутствие сотрудника',
  employee_presence: 'Появление сотрудника',
  visitor_shelf_dwell: 'Долгое нахождение у полки',
  hand_to_body: 'Рука к телу/сумке',
  back_to_camera: 'Спиной к камере',
  system_stream_lost: 'Проблема RTSP',
}

const STATUS_LABELS: Record<EventStatus, string> = {
  new: 'Новое',
  confirmed: 'Подтверждено',
  dismissed: 'Отклонено',
}

const ROLE_LABELS: Record<Role, string> = {
  admin: 'Админ',
  operator: 'Оператор',
  manager: 'Руководитель',
}

const emptyFilters: Filters = {
  cameraId: '',
  type: '',
  status: '',
  dateFrom: '',
  dateTo: '',
}

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers)
  headers.set('Content-Type', 'application/json')
  if (API_AUTH) {
    headers.set('Authorization', API_AUTH)
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const details = await response.text()
    throw new Error(details || `HTTP ${response.status}`)
  }

  return (await response.json()) as T
}

function buildEventQuery(filters: Filters) {
  const params = new URLSearchParams()
  if (filters.cameraId) params.set('camera_id', filters.cameraId)
  if (filters.type) params.set('event_type', filters.type)
  if (filters.status) params.set('status', filters.status)
  if (filters.dateFrom) params.set('date_from', new Date(filters.dateFrom).toISOString())
  if (filters.dateTo) params.set('date_to', new Date(filters.dateTo).toISOString())
  const query = params.toString()
  return query ? `/api/events?${query}` : '/api/events'
}

function reportHref(kind: 'csv' | 'pdf') {
  return `${API_URL}/api/reports/day.${kind}`
}

function snapshotSrc(event: VideoEvent) {
  if (event.snapshot_url.startsWith('http')) return event.snapshot_url
  return `${API_URL}${event.snapshot_url}`
}

function cameraLiveSrc(camera: MonitoringCamera) {
  return `${API_URL}/api/cameras/${camera.id}/live.svg`
}

function severityClass(severity: Severity) {
  if (severity === 'high') return 'border-red-400 bg-red-50 text-red-700'
  if (severity === 'medium') return 'border-amber-400 bg-amber-50 text-amber-700'
  return 'border-emerald-400 bg-emerald-50 text-emerald-700'
}

function statusClass(status: EventStatus) {
  if (status === 'confirmed') return 'bg-emerald-100 text-emerald-700'
  if (status === 'dismissed') return 'bg-slate-100 text-slate-600'
  return 'bg-blue-100 text-blue-700'
}

function cameraStatusClass(status: MonitoringCamera['status']) {
  if (status === 'online') return 'bg-emerald-400'
  if (status === 'unstable') return 'bg-amber-400'
  return 'bg-red-400'
}

function roleClass(role: Role) {
  if (role === 'admin') return 'bg-violet-100 text-violet-700'
  if (role === 'manager') return 'bg-amber-100 text-amber-700'
  return 'bg-cyan-100 text-cyan-700'
}

function formatSeconds(seconds?: number | null) {
  if (!seconds || seconds <= 0) return '—'
  if (seconds < 60) return `${Math.round(seconds)} сек.`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} мин.`
  return `${Math.round(minutes / 60)} ч.`
}

function qualityLabel(score?: number) {
  if (score === undefined) return '—'
  return `${Math.round(score > 1 ? score : score * 100)}%`
}

function App() {
  const [overview, setOverview] = useState<Overview | null>(null)
  const [analytics, setAnalytics] = useState<ShiftAnalytics | null>(null)
  const [cameras, setCameras] = useState<MonitoringCamera[]>([])
  const [events, setEvents] = useState<VideoEvent[]>([])
  const [settings, setSettings] = useState<MonitoringSettings | null>(null)
  const [telegramPreview, setTelegramPreview] = useState<TelegramPreview | null>(null)
  const [publicSources, setPublicSources] = useState<PublicVideoSource[]>([])
  const [capabilities, setCapabilities] = useState<DetectionCapability[]>([])
  const [filters, setFilters] = useState<Filters>(emptyFilters)
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [role, setRole] = useState<Role>('operator')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const loadEvents = useCallback(async (nextFilters: Filters) => {
    const eventData = await fetchJson<VideoEvent[]>(buildEventQuery(nextFilters))
    setEvents(eventData)
    return eventData
  }, [])

  const loadDashboard = useCallback(async () => {
    setError(null)
    const [overviewData, cameraData, analyticsData, settingsData, eventData, sourceData, capabilityData] = await Promise.all([
      fetchJson<Overview>('/api/overview'),
      fetchJson<MonitoringCamera[]>('/api/cameras'),
      fetchJson<ShiftAnalytics>('/api/shift/analytics'),
      fetchJson<MonitoringSettings>('/api/settings'),
      fetchJson<VideoEvent[]>(buildEventQuery(filters)),
      fetchJson<PublicVideoSource[]>('/api/public-sources'),
      fetchJson<DetectionCapability[]>('/api/detection-capabilities'),
    ])
    setOverview(overviewData)
    setCameras(cameraData)
    setAnalytics(analyticsData)
    setSettings(settingsData)
    setEvents(eventData)
    setPublicSources(sourceData)
    setCapabilities(capabilityData)

    if (eventData.length > 0) {
      const preview = await fetchJson<TelegramPreview>(`/api/telegram/preview?event_id=${eventData[0].id}`)
      setTelegramPreview(preview)
    }
  }, [filters])

  useEffect(() => {
    const load = async () => {
      try {
        await loadDashboard()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Не удалось загрузить dashboard')
      } finally {
        setLoading(false)
      }
    }

    void load()
  }, [loadDashboard])

  const suspiciousRate = useMemo(() => {
    if (overview === null || analytics === null || analytics.total_events === 0) return 0
    return Math.round((overview.suspicious_events / analytics.total_events) * 100)
  }, [analytics, overview])

  const handleRefresh = async () => {
    setBusy(true)
    try {
      await loadDashboard()
      setNotice('Данные обновлены')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка обновления')
    } finally {
      setBusy(false)
    }
  }

  const handleApplyFilters = async () => {
    setBusy(true)
    try {
      await loadEvents(filters)
      setNotice('Фильтры применены')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка фильтрации')
    } finally {
      setBusy(false)
    }
  }

  const handleResetFilters = async () => {
    setBusy(true)
    try {
      setFilters(emptyFilters)
      await loadEvents(emptyFilters)
      setNotice('Фильтры сброшены')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка сброса фильтров')
    } finally {
      setBusy(false)
    }
  }

  const handleSimulate = async () => {
    setBusy(true)
    try {
      const event = await fetchJson<VideoEvent>('/api/events/simulate', {
        method: 'POST',
        body: JSON.stringify({ camera_id: 'cam-sales-floor-demo' }),
      })
      setFilters(emptyFilters)
      await loadDashboard()
      setNotice(`Событие создано: ${event.title}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка симуляции события')
    } finally {
      setBusy(false)
    }
  }

  const handleFeedback = async (eventId: string, status: EventStatus) => {
    setBusy(true)
    try {
      await fetchJson<VideoEvent>(`/api/events/${eventId}/feedback`, {
        method: 'POST',
        body: JSON.stringify({ status, reviewed_by: ROLE_LABELS[role], note: 'Feedback from dashboard' }),
      })
      await loadDashboard()
      setNotice(status === 'confirmed' ? 'Событие подтверждено' : 'Событие отклонено')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка отправки feedback')
    } finally {
      setBusy(false)
    }
  }

  const handleTelegramTest = async () => {
    setBusy(true)
    try {
      const result = await fetchJson<TelegramTestResponse>('/api/telegram/test', { method: 'POST' })
      setTelegramPreview(result.preview)
      setNotice(result.configured ? result.detail : 'Telegram не настроен, показан mock-preview с inline-кнопками')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка Telegram test')
    } finally {
      setBusy(false)
    }
  }

  const handleSaveSettings = async () => {
    if (settings === null) return
    setBusy(true)
    try {
      const nextSettings = await fetchJson<MonitoringSettings>('/api/settings', {
        method: 'PUT',
        body: JSON.stringify(settings),
      })
      setSettings(nextSettings)
      setNotice('Пороги сохранены')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка сохранения настроек')
    } finally {
      setBusy(false)
    }
  }

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-950 text-white">
        <div className="flex items-center gap-3 rounded-2xl bg-white/10 px-6 py-4 shadow-2xl">
          <RefreshCw className="animate-spin" size={22} />
          <span>Загрузка AI Video Monitoring MVP...</span>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-6 py-8">
        <header className="flex flex-col gap-5 rounded-3xl border border-white/10 bg-white/10 p-6 shadow-2xl backdrop-blur lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="mb-2 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-cyan-300">
              <ShieldAlert size={18} /> MVP AI-видеомониторинга v2
            </p>
            <h1 className="text-3xl font-bold tracking-tight text-white lg:text-5xl">
              Камеры, события, отчёты и feedback
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300 lg:text-base">
              Прототип показывает почти пилотный контур: public research/demo video sources, mock RTSP,
              live preview с bbox/зонами, SQLite persistence, кадры к событиям, аналитику смены и Telegram inline workflow.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className={`inline-flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-bold ${roleClass(role)}`}>
              <Users size={18} /> {ROLE_LABELS[role]}
            </div>
            <select
              className="rounded-xl border border-white/20 bg-slate-900 px-4 py-3 text-sm font-bold text-white"
              value={role}
              onChange={(event) => setRole(event.target.value as Role)}
            >
              <option value="operator">Оператор</option>
              <option value="manager">Руководитель</option>
              <option value="admin">Админ</option>
            </select>
            <button
              className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-4 py-3 text-sm font-bold text-slate-950 shadow-lg transition hover:bg-cyan-300 disabled:opacity-60"
              disabled={busy}
              onClick={handleSimulate}
            >
              <Activity size={18} /> Симулировать событие
            </button>
            <button
              className="inline-flex items-center gap-2 rounded-xl border border-white/20 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/10 disabled:opacity-60"
              disabled={busy}
              onClick={handleRefresh}
            >
              <RefreshCw className={busy ? 'animate-spin' : ''} size={18} /> Обновить
            </button>
          </div>
        </header>

        <nav className="flex flex-wrap gap-2 rounded-3xl border border-white/10 bg-white/10 p-2">
          <TabButton active={activeTab === 'overview'} icon={<BarChart3 size={18} />} label="Обзор" onClick={() => setActiveTab('overview')} />
          <TabButton active={activeTab === 'events'} icon={<ShieldAlert size={18} />} label="События" onClick={() => setActiveTab('events')} />
          <TabButton active={activeTab === 'analytics'} icon={<FileText size={18} />} label="Аналитика и отчёты" onClick={() => setActiveTab('analytics')} />
          <TabButton active={activeTab === 'cameras'} icon={<Camera size={18} />} label="Камеры" onClick={() => setActiveTab('cameras')} />
          <TabButton active={activeTab === 'settings'} icon={<Settings size={18} />} label="Настройки" onClick={() => setActiveTab('settings')} />
        </nav>

        {error !== null && (
          <div className="rounded-2xl border border-red-400 bg-red-950 px-5 py-4 text-red-100">
            {error}
          </div>
        )}
        {notice !== null && (
          <div className="rounded-2xl border border-emerald-400 bg-emerald-950 px-5 py-4 text-emerald-100">
            {notice}
          </div>
        )}

        {(activeTab === 'overview' || activeTab === 'analytics') && (
          <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard icon={<Camera size={24} />} label="Камеры online" value={`${analytics?.cameras_online ?? overview?.active_cameras ?? 0}/${analytics?.cameras_total ?? overview?.total_cameras ?? 0}`} hint="Public demo + mock RTSP потоки" />
            <MetricCard icon={<ShieldAlert size={24} />} label="Новые события" value={analytics?.open_events ?? overview?.open_events ?? 0} hint="Ожидают проверки" />
            <MetricCard icon={<UserCheck size={24} />} label="Отсутствия" value={analytics?.absence_events ?? overview?.absence_events ?? 0} hint="Контроль рабочих зон" />
            <MetricCard icon={<Activity size={24} />} label="Подозрительные" value={`${suspiciousRate}%`} hint="Доля эвристик среди событий" />
          </section>
        )}

        {activeTab === 'overview' && (
          <section className="grid gap-6 lg:grid-cols-2">
            <LiveCameraPanel cameras={cameras.slice(0, 3)} sources={publicSources} />
            <CapabilityPanel capabilities={capabilities} />
          </section>
        )}

        {activeTab === 'overview' && (
          <section className="grid gap-6 lg:grid-cols-3">
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl lg:col-span-2">
              <SectionTitle icon={<ShieldAlert size={22} />} title="Последние события с кадрами" subtitle="Snapshot, confidence, статус и быстрый feedback оператора." />
              <div className="space-y-4">
                {events.slice(0, 3).map((event) => (
                  <EventCard key={event.id} event={event} busy={busy} role={role} onFeedback={handleFeedback} />
                ))}
              </div>
            </div>
            <div className="space-y-6">
              <TelegramPreviewCard preview={telegramPreview} busy={busy} onTest={handleTelegramTest} />
              <ReportsCard />
            </div>
          </section>
        )}

        {activeTab === 'events' && (
          <section className="grid gap-6 lg:grid-cols-4">
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl lg:col-span-1">
              <SectionTitle icon={<SlidersHorizontal size={22} />} title="Фильтры" subtitle="Камера, тип, статус и период." />
              <div className="mt-5 space-y-4">
                <Field label="Камера">
                  <select className="input" value={filters.cameraId} onChange={(event) => setFilters({ ...filters, cameraId: event.target.value })}>
                    <option value="">Все камеры</option>
                    {cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name}</option>)}
                  </select>
                </Field>
                <Field label="Тип события">
                  <select className="input" value={filters.type} onChange={(event) => setFilters({ ...filters, type: event.target.value })}>
                    <option value="">Все типы</option>
                    {Object.entries(EVENT_TYPE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </Field>
                <Field label="Статус">
                  <select className="input" value={filters.status} onChange={(event) => setFilters({ ...filters, status: event.target.value })}>
                    <option value="">Все статусы</option>
                    {Object.entries(STATUS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                </Field>
                <Field label="С даты">
                  <input className="input" type="datetime-local" value={filters.dateFrom} onChange={(event) => setFilters({ ...filters, dateFrom: event.target.value })} />
                </Field>
                <Field label="По дату">
                  <input className="input" type="datetime-local" value={filters.dateTo} onChange={(event) => setFilters({ ...filters, dateTo: event.target.value })} />
                </Field>
                <button className="btn-primary w-full" disabled={busy} onClick={handleApplyFilters}>Применить</button>
                <button className="btn-secondary w-full" disabled={busy} onClick={handleResetFilters}>Сбросить</button>
              </div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl lg:col-span-3">
              <SectionTitle icon={<ShieldAlert size={22} />} title="Лента событий" subtitle="События-кандидаты требуют подтверждения или отклонения." />
              <div className="mt-5 space-y-4">
                {events.map((event) => (
                  <EventCard key={event.id} event={event} busy={busy} role={role} onFeedback={handleFeedback} />
                ))}
              </div>
            </div>
          </section>
        )}

        {activeTab === 'analytics' && analytics !== null && (
          <section className="grid gap-6 lg:grid-cols-3">
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl lg:col-span-2">
              <SectionTitle icon={<BarChart3 size={22} />} title="Журнал смены / дня" subtitle={`Дата отчёта: ${analytics.report_date}`} />
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <AnalyticsItem label="Всего событий" value={analytics.total_events} />
                <AnalyticsItem label="Подтверждено" value={analytics.confirmed_events} />
                <AnalyticsItem label="Отклонено" value={analytics.dismissed_events} />
                <AnalyticsItem label="Новые" value={analytics.open_events} />
                <AnalyticsItem label="Отсутствия" value={analytics.absence_events} />
                <AnalyticsItem label="Подозрительные" value={analytics.suspicious_events} />
                <AnalyticsItem label="Среднее время реакции" value={formatSeconds(analytics.average_reaction_seconds)} />
                <AnalyticsItem label="Telegram" value={analytics.telegram_configured ? 'Настроен' : 'Mock-mode'} />
              </div>
            </div>
            <div className="space-y-6">
              <ReportsCard />
              <TelegramPreviewCard preview={telegramPreview} busy={busy} onTest={handleTelegramTest} />
            </div>
          </section>
        )}

        {activeTab === 'cameras' && (
          <section className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
            <SectionTitle icon={<Camera size={22} />} title="Страница камер" subtitle="RTSP URL, статус потока, FPS, качество, AI status, последнее событие и uptime." />
            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              {cameras.map((camera) => (
                <article key={camera.id} className="rounded-2xl border border-slate-200 p-5">
                  <div className="mb-4 flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-lg font-bold">{camera.name}</h3>
                      <p className="text-sm text-slate-500">{camera.location}</p>
                    </div>
                    <span className={`mt-1 h-3 w-3 rounded-full ${cameraStatusClass(camera.status)}`} />
                  </div>
                  <img className="mb-4 w-full rounded-2xl border border-slate-200 bg-slate-950 object-cover" src={cameraLiveSrc(camera)} alt={`Live preview ${camera.name}`} />
                  <div className="grid gap-3 text-sm sm:grid-cols-2">
                    <CameraField label="Источник" value={camera.rtsp_url} />
                    <CameraField label="Тип источника" value={camera.source_type ?? 'mock_rtsp'} />
                    <CameraField label="Статус" value={camera.status} />
                    <CameraField label="AI status" value={camera.ai_status} />
                    <CameraField label="FPS" value={camera.fps.toString()} />
                    <CameraField label="Качество" value={qualityLabel(camera.quality_score)} />
                    <CameraField label="Uptime" value={formatSeconds((camera.uptime_minutes ?? 0) * 60)} />
                    <CameraField label="Последнее событие" value={camera.last_event_at ? new Date(camera.last_event_at).toLocaleString('ru-RU') : camera.last_event_title ?? '—'} />
                    <CameraField label="Последний кадр" value={new Date(camera.last_seen_at).toLocaleString('ru-RU')} />
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {camera.zones.map((zone) => (
                      <span key={zone.id} className="rounded-lg bg-slate-900 px-2 py-1 text-xs text-white">
                        {zone.name}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'settings' && settings !== null && (
          <section className="grid gap-6 lg:grid-cols-3">
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl lg:col-span-2">
              <SectionTitle icon={<Settings size={22} />} title="Настройки порогов" subtitle="Простая адаптация MVP под торговую точку клиента." />
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                <Field label="Отсутствие сотрудника, мин.">
                  <input className="input" min={1} type="number" value={settings.absence_threshold_minutes} onChange={(event) => setSettings({ ...settings, absence_threshold_minutes: Number(event.target.value) })} />
                </Field>
                <Field label="У полки, сек.">
                  <input className="input" min={5} type="number" value={settings.shelf_dwell_seconds} onChange={(event) => setSettings({ ...settings, shelf_dwell_seconds: Number(event.target.value) })} />
                </Field>
                <Field label="Confidence threshold">
                  <input className="input" max={1} min={0.1} step={0.05} type="number" value={settings.confidence_threshold} onChange={(event) => setSettings({ ...settings, confidence_threshold: Number(event.target.value) })} />
                </Field>
              </div>
              <button className="btn-primary mt-5" disabled={busy} onClick={handleSaveSettings}>Сохранить настройки</button>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
              <SectionTitle icon={<Users size={22} />} title="Роли" subtitle="Визуальные бейджи без полноценной авторизации." />
              <div className="mt-5 space-y-3">
                {(['operator', 'manager', 'admin'] as Role[]).map((nextRole) => (
                  <button key={nextRole} className={`w-full rounded-xl px-4 py-3 text-left text-sm font-bold ${role === nextRole ? roleClass(nextRole) : 'bg-slate-100 text-slate-600'}`} onClick={() => setRole(nextRole)}>
                    {ROLE_LABELS[nextRole]}
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  )
}

type TabButtonProps = {
  active: boolean
  icon: ReactNode
  label: string
  onClick: () => void
}

function TabButton({ active, icon, label, onClick }: TabButtonProps) {
  return (
    <button className={`inline-flex items-center gap-2 rounded-2xl px-4 py-3 text-sm font-bold transition ${active ? 'bg-white text-slate-950' : 'text-slate-200 hover:bg-white/10'}`} onClick={onClick}>
      {icon} {label}
    </button>
  )
}

type SectionTitleProps = {
  icon: ReactNode
  title: string
  subtitle: string
}

function SectionTitle({ icon, title, subtitle }: SectionTitleProps) {
  return (
    <div className="flex items-start gap-3">
      <div className="rounded-2xl bg-slate-950 p-3 text-cyan-300">{icon}</div>
      <div>
        <h2 className="text-xl font-bold">{title}</h2>
        <p className="text-sm text-slate-500">{subtitle}</p>
      </div>
    </div>
  )
}

type MetricCardProps = {
  icon: ReactNode
  label: string
  value: string | number
  hint: string
}

function MetricCard({ icon, label, value, hint }: MetricCardProps) {
  return (
    <article className="rounded-3xl border border-white/10 bg-white p-5 text-slate-950 shadow-xl">
      <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-950 text-cyan-300">
        {icon}
      </div>
      <p className="text-sm font-semibold text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-black">{value}</p>
      <p className="mt-2 text-sm text-slate-500">{hint}</p>
    </article>
  )
}

type EventCardProps = {
  event: VideoEvent
  busy: boolean
  role: Role
  onFeedback: (eventId: string, status: EventStatus) => Promise<void>
}

function EventCard({ event, busy, role, onFeedback }: EventCardProps) {
  return (
    <article className="rounded-2xl border border-slate-200 p-4">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start">
        <img className="h-44 w-full rounded-2xl border border-slate-200 object-cover xl:w-72" src={snapshotSrc(event)} alt={event.title} />
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap gap-2">
            <span className={`rounded-full px-3 py-1 text-xs font-bold ${statusClass(event.status)}`}>{STATUS_LABELS[event.status]}</span>
            <span className={`rounded-full border px-3 py-1 text-xs font-bold ${severityClass(event.severity)}`}>{EVENT_TYPE_LABELS[event.type]}</span>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600">{Math.round(event.confidence * 100)}% confidence</span>
            <span className={`rounded-full px-3 py-1 text-xs font-bold ${roleClass(role)}`}>{ROLE_LABELS[role]}</span>
          </div>
          <h3 className="text-lg font-bold">{event.title}</h3>
          <p className="mt-1 text-sm leading-6 text-slate-600">{event.description}</p>
          <div className="mt-3 rounded-xl bg-cyan-50 p-3 text-sm text-cyan-950">
            <p className="font-bold">Анализ признаков</p>
            <p className="mt-1 leading-5">{event.analysis_summary}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {event.evidence_tags.map((tag) => <span key={tag} className="rounded-lg bg-white px-2 py-1 text-xs font-bold text-cyan-700">{tag}</span>)}
            </div>
          </div>
          <p className="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            {event.camera_name} • {event.zone} • {new Date(event.detected_at).toLocaleString('ru-RU')}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <button className="inline-flex items-center gap-2 rounded-xl bg-emerald-100 px-3 py-2 text-sm font-bold text-emerald-700 transition hover:bg-emerald-200 disabled:opacity-60" disabled={busy || event.status === 'confirmed'} onClick={() => void onFeedback(event.id, 'confirmed')}>
              <CheckCircle2 size={17} /> Подтвердить
            </button>
            <button className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-3 py-2 text-sm font-bold text-slate-600 transition hover:bg-slate-200 disabled:opacity-60" disabled={busy || event.status === 'dismissed'} onClick={() => void onFeedback(event.id, 'dismissed')}>
              <XCircle size={17} /> Отклонить
            </button>
          </div>
        </div>
      </div>
    </article>
  )
}

type FieldProps = {
  label: string
  children: ReactNode
}

function Field({ label, children }: FieldProps) {
  return (
    <label className="block text-sm font-bold text-slate-600">
      <span className="mb-2 block">{label}</span>
      {children}
    </label>
  )
}

type AnalyticsItemProps = {
  label: string
  value: string | number
}

function AnalyticsItem({ label, value }: AnalyticsItemProps) {
  return (
    <div className="rounded-2xl bg-slate-100 p-5">
      <p className="text-sm font-semibold text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-black text-slate-950">{value}</p>
    </div>
  )
}

type CameraFieldProps = {
  label: string
  value: string
}

function CameraField({ label, value }: CameraFieldProps) {
  return (
    <div className="rounded-xl bg-slate-100 p-3">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 break-words font-semibold text-slate-800">{value}</p>
    </div>
  )
}

type LiveCameraPanelProps = {
  cameras: MonitoringCamera[]
  sources: PublicVideoSource[]
}

function LiveCameraPanel({ cameras, sources }: LiveCameraPanelProps) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
      <SectionTitle icon={<Camera size={22} />} title="Public/demo live камеры" subtitle="Безопасные research/demo источники вместо сомнительных открытых RTSP." />
      <div className="mt-5 space-y-4">
        {cameras.map((camera) => (
          <article key={camera.id} className="rounded-2xl border border-slate-200 p-3">
            <img className="w-full rounded-xl border border-slate-200 bg-slate-950 object-cover" src={cameraLiveSrc(camera)} alt={camera.name} />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="font-bold">{camera.name}</p>
                <p className="text-sm text-slate-500">{camera.source_type ?? 'mock_rtsp'} • FPS {camera.fps} • quality {qualityLabel(camera.quality_score)}</p>
              </div>
              <span className="rounded-full bg-cyan-100 px-3 py-1 text-xs font-bold text-cyan-700">bbox + zones</span>
            </div>
          </article>
        ))}
      </div>
      <div className="mt-5 space-y-2">
        {sources.map((source) => (
          <a key={source.id} className="block rounded-xl bg-slate-100 p-3 text-sm transition hover:bg-slate-200" href={source.source_url} target="_blank" rel="noreferrer">
            <span className="font-bold text-slate-950">{source.title}</span>
            <span className="mt-1 block text-slate-600">{source.scenario}</span>
          </a>
        ))}
      </div>
    </div>
  )
}

type CapabilityPanelProps = {
  capabilities: DetectionCapability[]
}

function readinessLabel(readiness: DetectionCapability['readiness']) {
  if (readiness === 'demo_ready') return 'Demo-ready'
  if (readiness === 'heuristic_ready') return 'Эвристика готова'
  return 'Нужен pilot'
}

function CapabilityPanel({ capabilities }: CapabilityPanelProps) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
      <SectionTitle icon={<BarChart3 size={22} />} title="Что уже умеем выявлять по ТЗ" subtitle="Признаки, доказательства, confidence и честные ограничения MVP." />
      <div className="mt-5 space-y-3">
        {capabilities.map((capability) => (
          <article key={capability.id} className="rounded-2xl border border-slate-200 p-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h3 className="font-bold">{capability.title}</h3>
                <p className="mt-1 text-sm text-slate-600">{capability.what_it_checks}</p>
              </div>
              <span className="rounded-full bg-slate-950 px-3 py-1 text-xs font-bold text-cyan-300">{Math.round(capability.confidence * 100)}%</span>
            </div>
            <div className="mt-3 flex flex-wrap gap-1">
              {capability.evidence.map((item) => <span key={item} className="rounded-lg bg-cyan-50 px-2 py-1 text-xs font-bold text-cyan-700">{item}</span>)}
            </div>
            <p className="mt-3 text-xs font-bold uppercase tracking-wide text-slate-400">{readinessLabel(capability.readiness)} • {capability.tz_mapping}</p>
            <p className="mt-2 text-sm text-slate-500">Ограничение: {capability.current_limitations}</p>
          </article>
        ))}
      </div>
    </div>
  )
}

type TelegramPreviewCardProps = {
  preview: TelegramPreview | null
  busy: boolean
  onTest: () => Promise<void>
}

function TelegramPreviewCard({ preview, busy, onTest }: TelegramPreviewCardProps) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
      <SectionTitle icon={<Bot size={22} />} title="Telegram inline-preview" subtitle="Mock или реальная отправка через env token." />
      <div className="mt-5 rounded-2xl bg-slate-950 p-4 text-slate-100">
        <p className="whitespace-pre-line text-sm leading-6">{preview?.text ?? 'Нет события для preview'}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {(preview?.buttons ?? []).map((button) => (
            <span key={button.callback_data} className="rounded-xl bg-cyan-400 px-3 py-2 text-xs font-bold text-slate-950">{button.label}</span>
          ))}
        </div>
      </div>
      <button className="btn-primary mt-5 w-full" disabled={busy} onClick={() => void onTest()}>
        <Bell size={18} /> Telegram test
      </button>
    </div>
  )
}

function ReportsCard() {
  return (
    <div className="rounded-3xl border border-white/10 bg-white p-6 text-slate-950 shadow-xl">
      <SectionTitle icon={<Download size={22} />} title="Отчёты" subtitle="Экспорт отчёта дня для руководителя." />
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <a className="btn-primary justify-center" href={reportHref('csv')} target="_blank" rel="noreferrer">
          CSV отчёт
        </a>
        <a className="btn-secondary justify-center" href={reportHref('pdf')} target="_blank" rel="noreferrer">
          PDF отчёт
        </a>
      </div>
    </div>
  )
}

export default App
