import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  ShieldAlert,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Inbox,
  Activity,
  Users,
  ListChecks,
  XCircle,
  ThumbsUp,
  ThumbsDown,
  ChevronRight,
  Loader2,
  Flag,
  Info,
} from 'lucide-react';

/* ─── Helpers ─────────────────────────────────────────── */
function riskColor(score) {
  if (score >= 0.8) return { badge: 'bg-red-900/50 border-red-700/50 text-red-300', bar: 'bg-red-500', label: 'High Risk' };
  if (score >= 0.4) return { badge: 'bg-amber-900/50 border-amber-700/50 text-amber-300', bar: 'bg-amber-500', label: 'Medium Risk' };
  return { badge: 'bg-green-900/50 border-green-700/50 text-green-300', bar: 'bg-green-500', label: 'Low Risk' };
}

function actionColor(action) {
  if (action === 'HARD_REJECT')   return 'bg-red-900/50 border-red-700/50 text-red-300';
  if (action === 'MANUAL_REVIEW') return 'bg-amber-900/50 border-amber-700/50 text-amber-300';
  if (action === 'CLEARED')       return 'bg-green-900/50 border-green-700/50 text-green-300';
  return 'bg-zinc-800 border-zinc-700 text-zinc-300';
}

function formatTs(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return ts;
  }
}

/* ─── Toast ───────────────────────────────────────────── */
function Toast({ toasts }) {
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-center gap-3 px-4 py-3 rounded-xl border shadow-2xl text-sm font-medium
            animate-[slideUp_0.3s_ease]
            ${t.type === 'success' ? 'bg-green-900/90 border-green-700 text-green-200'
              : t.type === 'error'  ? 'bg-red-900/90 border-red-700 text-red-200'
              : 'bg-zinc-800/90 border-zinc-700 text-zinc-200'}`}
        >
          {t.type === 'success' ? <CheckCircle2 size={16} /> : <Info size={16} />}
          {t.message}
        </div>
      ))}
    </div>
  );
}

/* ─── Stat Card ───────────────────────────────────────── */
function StatCard({ label, value, icon: Icon, color }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg ${color}`}>
        <Icon size={18} className="text-white" />
      </div>
      <div>
        <p className="text-2xl font-bold text-white leading-none">{value ?? 0}</p>
        <p className="text-xs text-zinc-500 mt-0.5">{label}</p>
      </div>
    </div>
  );
}

/* ─── List Item Card ──────────────────────────────────── */
function ListCard({ item, isSelected, onClick }) {
  const risk  = riskColor(item.fraud_risk_score ?? 0);
  const score = Math.round((item.fraud_risk_score ?? 0) * 100);

  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-xl border p-4 transition-all duration-150
        ${isSelected
          ? 'border-zomato-red bg-red-950/20 ring-1 ring-zomato-red/40'
          : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900'
        }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="font-semibold text-white text-sm truncate">{item.order_id ?? '—'}</p>
          <p className="text-xs text-zinc-500 mt-0.5 truncate">Customer: {item.customer_id ?? '—'}</p>
        </div>
        <span className={`flex-shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full border ${risk.badge}`}>
          {score}% risk
        </span>
      </div>

      {/* Flags */}
      {item.flags && item.flags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {item.flags.slice(0, 3).map((f, i) => (
            <span
              key={i}
              className="text-[10px] bg-zinc-800 border border-zinc-700 text-zinc-400 px-1.5 py-0.5 rounded-full truncate max-w-[140px]"
            >
              {f}
            </span>
          ))}
          {item.flags.length > 3 && (
            <span className="text-[10px] bg-zinc-800 border border-zinc-700 text-zinc-500 px-1.5 py-0.5 rounded-full">
              +{item.flags.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Timestamp + action */}
      <div className="mt-2 flex items-center justify-between">
        <span className="text-[10px] text-zinc-600">{formatTs(item.timestamp ?? item.created_at)}</span>
        {item.action && (
          <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${actionColor(item.action)}`}>
            {item.action}
          </span>
        )}
      </div>
    </button>
  );
}

/* ─── Detail Panel ────────────────────────────────────── */
function DetailPanel({ item, onApprove, onReject }) {
  if (!item) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <div className="p-4 bg-zinc-900 rounded-full mb-3">
          <Activity size={28} className="text-zinc-600" />
        </div>
        <p className="text-zinc-500 text-sm font-medium">Select an item to view details</p>
        <p className="text-zinc-700 text-xs mt-1">Click any card on the left</p>
      </div>
    );
  }

  const risk  = riskColor(item.fraud_risk_score ?? 0);
  const score = item.fraud_risk_score ?? 0;
  const pct   = Math.round(score * 100);

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 space-y-5 animate-[fadeIn_0.25s_ease]">
      {/* ID header */}
      <div>
        <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Order</p>
        <p className="text-xl font-bold text-white">{item.order_id ?? '—'}</p>
        <p className="text-sm text-zinc-500 mt-0.5">Customer: {item.customer_id ?? '—'}</p>
      </div>

      {/* Risk gauge */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-zinc-400">Fraud Risk Score</span>
          <span className={`text-sm font-bold ${score >= 0.8 ? 'text-red-400' : score >= 0.4 ? 'text-amber-400' : 'text-green-400'}`}>
            {pct}% — {risk.label}
          </span>
        </div>
        <div className="w-full h-3 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${risk.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Action badge */}
      {item.action && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Action Taken:</span>
          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${actionColor(item.action)}`}>
            {item.action}
          </span>
        </div>
      )}

      {/* Issue description */}
      {item.issue_description && (
        <div>
          <p className="text-xs font-medium text-zinc-500 mb-1.5">Issue Description</p>
          <p className="text-sm text-zinc-300 leading-relaxed bg-zinc-900 rounded-lg p-3 border border-zinc-800">
            {item.issue_description}
          </p>
        </div>
      )}

      {/* Fraud flags */}
      {item.flags && item.flags.length > 0 && (
        <div>
          <p className="text-xs font-medium text-zinc-500 mb-2 flex items-center gap-1.5">
            <Flag size={12} />
            Fraud Flags ({item.flags.length})
          </p>
          <ul className="space-y-1.5">
            {item.flags.map((flag, i) => {
              const isHigh = /high|critical|duplicate|multiple/i.test(flag);
              return (
                <li
                  key={i}
                  className={`flex items-start gap-2 text-xs rounded-lg px-3 py-2 border ${
                    isHigh
                      ? 'bg-red-950/30 border-red-900/50 text-red-300'
                      : 'bg-amber-950/20 border-amber-900/40 text-amber-300'
                  }`}
                >
                  <span className="flex-shrink-0 mt-0.5">{isHigh ? '🚨' : '⚠️'}</span>
                  {flag}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Approve / Reject */}
      <div className="flex gap-3 pt-2 border-t border-zinc-800">
        <button
          onClick={() => onApprove(item)}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl
            bg-green-900/40 border border-green-700/50 text-green-300 text-sm font-medium
            hover:bg-green-900/60 hover:border-green-600 transition-all duration-150"
        >
          <ThumbsUp size={15} />
          Approve
        </button>
        <button
          onClick={() => onReject(item)}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl
            bg-red-900/30 border border-red-800/50 text-red-300 text-sm font-medium
            hover:bg-red-900/50 hover:border-red-700 transition-all duration-150"
        >
          <ThumbsDown size={15} />
          Reject
        </button>
      </div>
    </div>
  );
}

/* ─── Empty State ─────────────────────────────────────── */
function EmptyState({ label }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="p-5 bg-zinc-900 rounded-full mb-4">
        <Inbox size={32} className="text-zinc-600" />
      </div>
      <p className="text-zinc-400 font-medium">{label}</p>
      <p className="text-zinc-700 text-xs mt-1">Items will appear here when available</p>
    </div>
  );
}

/* ─── Main Component ──────────────────────────────────── */
export default function AdminDashboard() {
  const [queueItems,    setQueueItems]    = useState([]);
  const [suspiciousLog, setSuspiciousLog] = useState([]);
  const [selectedItem,  setSelectedItem]  = useState(null);
  const [activeSection, setActiveSection] = useState('queue');
  const [loading,       setLoading]       = useState(true);
  const [refreshing,    setRefreshing]    = useState(false);
  const [toasts,        setToasts]        = useState([]);

  /* Toast helper */
  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3500);
  }, []);

  /* Fetch data */
  const fetchData = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else         setRefreshing(true);

    try {
      const [queueRes, logRes] = await Promise.allSettled([
        fetch('/api/v1/admin/queue'),
        fetch('/api/v1/admin/suspicious-log'),
      ]);

      if (queueRes.status === 'fulfilled' && queueRes.value.ok) {
        const data = await queueRes.value.json();
        setQueueItems(Array.isArray(data) ? data : data.items ?? []);
      }

      if (logRes.status === 'fulfilled' && logRes.value.ok) {
        const data = await logRes.value.json();
        setSuspiciousLog(Array.isArray(data) ? data : data.items ?? []);
      }
    } catch (err) {
      console.error('Admin fetch error:', err);
      addToast('Failed to load data from server', 'error');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [addToast]);

  useEffect(() => { fetchData(false); }, [fetchData]);

  /* Computed stats */
  const hardRejectsToday = suspiciousLog.filter(
    (item) => item.action === 'HARD_REJECT'
  ).length;
  const manualReviewPending = queueItems.length;

  /* Handlers */
  const handleApprove = useCallback((item) => {
    addToast(`✅ Claim ${item.order_id} approved (visual only)`, 'success');
  }, [addToast]);

  const handleReject = useCallback((item) => {
    addToast(`❌ Claim ${item.order_id} rejected (visual only)`, 'error');
  }, [addToast]);

  const displayItems = activeSection === 'queue' ? queueItems : suspiciousLog;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* ── Header ── */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <ShieldAlert size={22} className="text-zomato-red" />
            <h1 className="text-2xl font-bold text-white">Admin Control Panel</h1>
          </div>
          <p className="text-zinc-500 text-sm mt-1">Fraud & Dispute Management</p>
        </div>
        <button
          onClick={() => fetchData(true)}
          disabled={refreshing || loading}
          className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700
            text-zinc-300 text-sm font-medium rounded-xl transition-all duration-150 disabled:opacity-50"
        >
          <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {/* ── Stats Bar ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Items in Queue"
          value={queueItems.length}
          icon={ListChecks}
          color="bg-zomato-red"
        />
        <StatCard
          label="Suspicious Log Entries"
          value={suspiciousLog.length}
          icon={Activity}
          color="bg-amber-600"
        />
        <StatCard
          label="Hard Rejects Today"
          value={hardRejectsToday}
          icon={XCircle}
          color="bg-red-700"
        />
        <StatCard
          label="Manual Reviews Pending"
          value={manualReviewPending}
          icon={Users}
          color="bg-blue-700"
        />
      </div>

      {/* ── Tab Switcher ── */}
      <div className="flex items-center gap-1 bg-zinc-900 rounded-xl p-1 w-fit mb-6">
        <button
          onClick={() => { setActiveSection('queue'); setSelectedItem(null); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
            activeSection === 'queue'
              ? 'bg-zomato-red text-white shadow-md'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
          }`}
        >
          <Clock size={14} />
          Review Queue
          <span className="bg-black/30 text-xs px-1.5 py-0.5 rounded-full">
            {queueItems.length}
          </span>
        </button>
        <button
          onClick={() => { setActiveSection('log'); setSelectedItem(null); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
            activeSection === 'log'
              ? 'bg-zomato-red text-white shadow-md'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
          }`}
        >
          <ShieldAlert size={14} />
          Suspicious Activity Log
          <span className="bg-black/30 text-xs px-1.5 py-0.5 rounded-full">
            {suspiciousLog.length}
          </span>
        </button>
      </div>

      {/* ── Main Two-Column Area ── */}
      {loading ? (
        <div className="flex items-center justify-center py-32">
          <div className="flex flex-col items-center gap-4">
            <Loader2 size={32} className="text-zomato-red animate-spin" />
            <p className="text-zinc-500 text-sm">Loading dashboard data…</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left pane — list */}
          <div className="lg:col-span-2 space-y-3">
            {displayItems.length === 0 ? (
              <EmptyState
                label={activeSection === 'queue' ? 'No items in review queue' : 'No suspicious activity logged'}
              />
            ) : (
              displayItems.map((item, idx) => (
                <ListCard
                  key={item.order_id ?? item.id ?? idx}
                  item={item}
                  isSelected={selectedItem && (selectedItem.order_id ?? selectedItem.id) === (item.order_id ?? item.id)}
                  onClick={() => setSelectedItem(item)}
                />
              ))
            )}
          </div>

          {/* Right pane — detail */}
          <div className="lg:col-span-3">
            <DetailPanel
              item={selectedItem}
              onApprove={handleApprove}
              onReject={handleReject}
            />
          </div>
        </div>
      )}

      {/* ── Toast Container ── */}
      <Toast toasts={toasts} />
    </div>
  );
}
