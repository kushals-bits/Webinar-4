import { useState, useRef, useCallback } from 'react';
import {
  Camera,
  Image as ImageIcon,
  X,
  Upload,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Clock,
  ShieldAlert,
  FileText,
  ChevronRight,
  Receipt,
  Utensils,
} from 'lucide-react';

/* ─── Helpers ─────────────────────────────────────────── */
const STEPS = [
  { key: 'analyzing_security',    label: 'Analyzing Security Profile', icon: '🔒' },
  { key: 'verifying_discrepancy', label: 'Verifying Discrepancy',       icon: '🔍' },
  { key: 'processing_refund',     label: 'Calculating Refund',          icon: '💰' },
  { key: 'completed',             label: 'Completed',                   icon: '✅' },
];

const TERMINAL_STATES = new Set(['completed', 'flagged', 'rejected', 'error']);

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = (e) => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ─── Sub-components ──────────────────────────────────── */
function DropZone({ label, icon: Icon, preview, dragActive, onDrop, onDragOver, onDragLeave, onFileChange, onRemove, inputRef }) {
  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed
        transition-all duration-200 cursor-pointer overflow-hidden min-h-[180px]
        ${dragActive
          ? 'border-zomato-red bg-red-950/30 scale-[1.01]'
          : 'border-zinc-700 bg-zinc-900/50 hover:border-zinc-500 hover:bg-zinc-900'
        }`}
      onClick={() => !preview && inputRef.current?.click()}
    >
      {preview ? (
        <>
          <img
            src={preview}
            alt={label}
            className="w-full h-full object-cover absolute inset-0 rounded-xl"
          />
          <div className="absolute inset-0 bg-black/40 rounded-xl" />
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            className="absolute top-2 right-2 z-10 bg-black/70 hover:bg-red-600 text-white rounded-full p-1 transition-colors"
          >
            <X size={14} />
          </button>
          <span className="relative z-10 text-xs text-white/80 font-medium bg-black/50 px-2 py-0.5 rounded-full mt-auto mb-2">
            {label}
          </span>
        </>
      ) : (
        <div className="flex flex-col items-center gap-3 p-6 text-center select-none">
          <div className={`p-3 rounded-full ${dragActive ? 'bg-red-800/40' : 'bg-zinc-800'}`}>
            <Icon size={22} className={dragActive ? 'text-zomato-red' : 'text-zinc-400'} />
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-300">{label}</p>
            <p className="text-xs text-zinc-500 mt-1">Drag & drop or click to browse</p>
            <p className="text-xs text-zinc-600 mt-0.5">PNG, JPG, WEBP up to 10 MB</p>
          </div>
          <span className="mt-1 inline-flex items-center gap-1.5 text-xs text-zinc-400 border border-zinc-700 rounded-lg px-3 py-1.5 hover:border-zinc-500 transition-colors">
            <Upload size={12} />
            Upload Image
          </span>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onFileChange}
      />
    </div>
  );
}

function StatusStepper({ submissionState }) {
  const activeIdx = STEPS.findIndex((s) => s.key === submissionState);
  const isTerminal = TERMINAL_STATES.has(submissionState);
  const isFailed = submissionState === 'rejected' || submissionState === 'flagged' || submissionState === 'error';

  if (submissionState === 'idle') return null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 animate-[fadeIn_0.3s_ease]">
      <p className="text-xs font-semibold text-zinc-500 uppercase tracking-widest mb-4">
        Live Processing Status
      </p>
      <div className="flex items-start gap-0">
        {STEPS.map((step, idx) => {
          const isPast    = isTerminal ? (isFailed ? idx < activeIdx : true) : idx < activeIdx;
          const isActive  = !isTerminal && idx === activeIdx;
          const isFuture  = !isTerminal && idx > activeIdx;
          const isFinalOk = isTerminal && !isFailed;

          const dotColor = isFailed && idx >= activeIdx
            ? 'bg-zinc-700'
            : isPast || isFinalOk
            ? 'bg-green-500'
            : isActive
            ? 'bg-zomato-red'
            : 'bg-zinc-700';

          const lineColor = isPast || isFinalOk ? 'bg-green-500/50' : 'bg-zinc-700';

          return (
            <div key={step.key} className="flex-1 flex flex-col items-center">
              {/* Dot + connector */}
              <div className="flex items-center w-full">
                {/* Left line */}
                {idx > 0 && (
                  <div className={`flex-1 h-0.5 ${lineColor} transition-colors duration-500`} />
                )}
                {/* Dot */}
                <div className="relative flex-shrink-0">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-sm
                      border-2 transition-all duration-500
                      ${isActive ? 'border-zomato-red bg-red-950 scale-110' : ''}
                      ${isPast || isFinalOk ? 'border-green-500 bg-green-900/40' : ''}
                      ${isFuture || (isFailed && idx > activeIdx) ? 'border-zinc-700 bg-zinc-800' : ''}
                    `}
                  >
                    {(isPast || isFinalOk) && !isActive ? (
                      <CheckCircle2 size={14} className="text-green-400" />
                    ) : (
                      <span className="text-xs">{step.icon}</span>
                    )}
                  </div>
                  {isActive && (
                    <span
                      className="status-pulse absolute -top-0.5 -right-0.5"
                      style={{ width: 8, height: 8 }}
                    />
                  )}
                </div>
                {/* Right line */}
                {idx < STEPS.length - 1 && (
                  <div
                    className={`flex-1 h-0.5 transition-colors duration-500 ${
                      isPast || isFinalOk ? 'bg-green-500/50' : 'bg-zinc-700'
                    }`}
                  />
                )}
              </div>

              {/* Label */}
              <p
                className={`mt-2 text-[10px] text-center font-medium leading-tight transition-colors duration-300 ${
                  isActive
                    ? 'text-zomato-red'
                    : isPast || isFinalOk
                    ? 'text-green-400'
                    : 'text-zinc-600'
                }`}
              >
                {step.label}
              </p>
              {isActive && (
                <Loader2 size={10} className="mt-1 text-zomato-red animate-spin" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ResultPanel({ result, submissionState }) {
  if (!result || submissionState === 'idle' || submissionState === 'analyzing_security' ||
      submissionState === 'verifying_discrepancy' || submissionState === 'processing_refund') {
    return null;
  }

  const { action, message, fraud_risk_score, refund_breakdown, total_refund, delivery_fee_waiver, flags } = result;

  const riskPct  = Math.round((fraud_risk_score ?? 0) * 100);
  const riskColor =
    riskPct >= 80 ? 'text-red-400'
    : riskPct >= 40 ? 'text-amber-400'
    : 'text-green-400';

  /* HARD_REJECT */
  if (action === 'HARD_REJECT' || submissionState === 'rejected' || submissionState === 'flagged') {
    return (
      <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-5 animate-[slideUp_0.4s_ease]">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-red-900/60 rounded-lg">
            <ShieldAlert size={20} className="text-red-400" />
          </div>
          <div>
            <p className="font-semibold text-red-300">🚫 Claim Rejected</p>
            <p className="text-xs text-red-500">Fraud detection triggered</p>
          </div>
          <span className={`ml-auto font-bold text-lg ${riskColor}`}>
            {riskPct}%
            <span className="text-xs font-normal text-zinc-500 ml-1">risk</span>
          </span>
        </div>
        <p className="text-sm text-red-200/80 leading-relaxed">{message}</p>
        {flags && flags.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {flags.map((f, i) => (
              <span key={i} className="text-xs bg-red-900/50 border border-red-800/40 text-red-300 px-2 py-1 rounded-full">
                🚨 {f}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  /* MANUAL_REVIEW */
  if (action === 'MANUAL_REVIEW') {
    return (
      <div className="bg-amber-950/30 border border-amber-700/50 rounded-xl p-5 animate-[slideUp_0.4s_ease]">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-amber-900/40 rounded-lg">
            <Clock size={20} className="text-amber-400" />
          </div>
          <div>
            <p className="font-semibold text-amber-300">⏳ Under Review</p>
            <p className="text-xs text-amber-600">Requires manual assessment</p>
          </div>
          <span className={`ml-auto font-bold text-lg ${riskColor}`}>
            {riskPct}%
            <span className="text-xs font-normal text-zinc-500 ml-1">risk</span>
          </span>
        </div>
        <p className="text-sm text-amber-200/80 leading-relaxed">{message}</p>
      </div>
    );
  }

  /* CLEARED — no refund */
  if (action === 'CLEARED' && (!total_refund || total_refund === 0)) {
    return (
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl p-5 animate-[slideUp_0.4s_ease]">
        <div className="flex items-center gap-3">
          <CheckCircle2 size={20} className="text-zinc-400" />
          <p className="text-zinc-300 font-medium">No Refund Applicable</p>
        </div>
        <p className="text-sm text-zinc-500 mt-2 leading-relaxed">{message}</p>
      </div>
    );
  }

  /* CLEARED — refund granted */
  return (
    <div className="bg-green-950/30 border border-green-800/50 rounded-xl p-5 animate-[slideUp_0.4s_ease]">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-green-900/40 rounded-lg">
          <CheckCircle2 size={20} className="text-green-400" />
        </div>
        <div>
          <p className="font-semibold text-green-300">✅ Refund Approved</p>
          <p className="text-xs text-green-700">Processing to original payment method</p>
        </div>
        <span className={`ml-auto text-xs font-semibold px-2.5 py-1 rounded-full border ${
          riskPct < 40
            ? 'bg-green-900/40 border-green-700 text-green-300'
            : 'bg-amber-900/40 border-amber-700 text-amber-300'
        }`}>
          Risk: {riskPct}%
        </span>
      </div>

      {/* Breakdown table */}
      {refund_breakdown && refund_breakdown.length > 0 && (
        <div className="bg-black/20 rounded-lg overflow-hidden mb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="text-left px-4 py-2 text-zinc-500 font-medium text-xs uppercase tracking-wide">
                  Item
                </th>
                <th className="text-right px-4 py-2 text-zinc-500 font-medium text-xs uppercase tracking-wide">
                  Amount
                </th>
              </tr>
            </thead>
            <tbody>
              {refund_breakdown.map((item, idx) => (
                <tr key={idx} className="border-b border-zinc-800/50 last:border-0">
                  <td className="px-4 py-2.5 text-zinc-200">{item.name ?? item.item ?? item[0]}</td>
                  <td className="px-4 py-2.5 text-right text-green-300 font-medium">
                    ₹{(item.amount ?? item[1] ?? 0).toFixed(2)}
                  </td>
                </tr>
              ))}
              {delivery_fee_waiver > 0 && (
                <tr className="border-b border-zinc-800/50">
                  <td className="px-4 py-2.5 text-zinc-200">🚚 Delivery Fee Waiver</td>
                  <td className="px-4 py-2.5 text-right text-green-300 font-medium">
                    ₹{Number(delivery_fee_waiver).toFixed(2)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Total */}
      <div className="flex items-baseline justify-between">
        <span className="text-zinc-400 text-sm font-medium">Total Refund</span>
        <span className="text-3xl font-black text-green-400">
          ₹{Number(total_refund).toFixed(2)}
        </span>
      </div>
      {message && (
        <p className="mt-3 text-xs text-green-700/80 border-t border-green-900/40 pt-3">{message}</p>
      )}
    </div>
  );
}

/* ─── Main Component ──────────────────────────────────── */
export default function CustomerClaimForm() {
  const [formData, setFormData]             = useState({ orderId: '', customerId: '', issueDescription: '' });
  const [billImage, setBillImage]           = useState(null);
  const [foodImage, setFoodImage]           = useState(null);
  const [billPreview, setBillPreview]       = useState(null);
  const [foodPreview, setFoodPreview]       = useState(null);
  const [submissionState, setSubmissionState] = useState('idle');
  const [result, setResult]                 = useState(null);
  const [dragOverBill, setDragOverBill]     = useState(false);
  const [dragOverFood, setDragOverFood]     = useState(false);
  const [errors, setErrors]                 = useState({});

  const billInputRef = useRef(null);
  const foodInputRef = useRef(null);

  /* File helpers */
  const handleFileSelect = useCallback(async (file, type) => {
    if (!file || !file.type.startsWith('image/')) return;
    const preview = await readFileAsDataURL(file);
    if (type === 'bill') {
      setBillImage(file);
      setBillPreview(preview);
    } else {
      setFoodImage(file);
      setFoodPreview(preview);
    }
  }, []);

  /* Drop zone handlers – bill */
  const onBillDrop = useCallback((e) => {
    e.preventDefault();
    setDragOverBill(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file, 'bill');
  }, [handleFileSelect]);

  const onBillDragOver = useCallback((e) => { e.preventDefault(); setDragOverBill(true); }, []);
  const onBillDragLeave = useCallback(() => setDragOverBill(false), []);
  const onBillFileChange = useCallback((e) => handleFileSelect(e.target.files[0], 'bill'), [handleFileSelect]);

  /* Drop zone handlers – food */
  const onFoodDrop = useCallback((e) => {
    e.preventDefault();
    setDragOverFood(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file, 'food');
  }, [handleFileSelect]);

  const onFoodDragOver = useCallback((e) => { e.preventDefault(); setDragOverFood(true); }, []);
  const onFoodDragLeave = useCallback(() => setDragOverFood(false), []);
  const onFoodFileChange = useCallback((e) => handleFileSelect(e.target.files[0], 'food'), [handleFileSelect]);

  /* Validation */
  const validate = () => {
    const errs = {};
    if (!formData.orderId.trim())         errs.orderId = 'Order ID is required';
    if (!formData.customerId.trim())      errs.customerId = 'Customer ID is required';
    if (!formData.issueDescription.trim()) errs.issueDescription = 'Please describe the issue';
    if (!billImage)                       errs.billImage = 'Bill photo is required';
    if (!foodImage)                       errs.foodImage = 'Food photo is required';
    return errs;
  };

  /* Submit */
  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setResult(null);

    const fd = new FormData();
    fd.append('order_id', formData.orderId);
    fd.append('customer_id', formData.customerId);
    fd.append('issue_description', formData.issueDescription);
    fd.append('bill_image', billImage);
    fd.append('food_image', foodImage);

    try {
      /* Step 1 */
      setSubmissionState('analyzing_security');
      await delay(1500);
      /* Step 2 */
      setSubmissionState('verifying_discrepancy');
      await delay(1500);
      /* Step 3 */
      setSubmissionState('processing_refund');

      const response = await fetch('/api/v1/claims/submit', {
        method: 'POST',
        body: fd,
      });

      await delay(800);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

      if (data.action === 'HARD_REJECT') {
        setSubmissionState('rejected');
      } else if (data.action === 'MANUAL_REVIEW') {
        setSubmissionState('flagged');
      } else {
        setSubmissionState('completed');
      }
    } catch (err) {
      console.error('Claim submission error:', err);
      setResult({ action: 'ERROR', message: err.message || 'An unexpected error occurred. Please try again.' });
      setSubmissionState('error');
    }
  };

  const isProcessing = !TERMINAL_STATES.has(submissionState) && submissionState !== 'idle';
  const canReset     = TERMINAL_STATES.has(submissionState);

  const handleReset = () => {
    setFormData({ orderId: '', customerId: '', issueDescription: '' });
    setBillImage(null);
    setFoodImage(null);
    setBillPreview(null);
    setFoodPreview(null);
    setSubmissionState('idle');
    setResult(null);
    setErrors({});
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <FileText size={20} className="text-zomato-red" />
          <h1 className="text-2xl font-bold text-white">File a Claim</h1>
        </div>
        <p className="text-zinc-500 text-sm">
          Submit your order dispute and our AI will assess your refund eligibility instantly.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6" noValidate>
        {/* ── Order Details Card ── */}
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-2 mb-1">
            <Receipt size={16} className="text-zinc-400" />
            <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wide">
              Order Details
            </h2>
          </div>

          {/* Order ID */}
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">
              Order ID <span className="text-zomato-red">*</span>
            </label>
            <input
              type="text"
              className={`input-field ${errors.orderId ? 'border-red-600 focus:ring-red-600' : ''}`}
              placeholder="ORD-2026-001"
              value={formData.orderId}
              onChange={(e) => setFormData((p) => ({ ...p, orderId: e.target.value }))}
              disabled={isProcessing}
            />
            {errors.orderId && (
              <p className="mt-1 text-xs text-red-500 flex items-center gap-1">
                <AlertTriangle size={11} /> {errors.orderId}
              </p>
            )}
          </div>

          {/* Customer ID */}
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">
              Customer ID <span className="text-zomato-red">*</span>
            </label>
            <input
              type="text"
              className={`input-field ${errors.customerId ? 'border-red-600 focus:ring-red-600' : ''}`}
              placeholder="CUST-12345"
              value={formData.customerId}
              onChange={(e) => setFormData((p) => ({ ...p, customerId: e.target.value }))}
              disabled={isProcessing}
            />
            {errors.customerId && (
              <p className="mt-1 text-xs text-red-500 flex items-center gap-1">
                <AlertTriangle size={11} /> {errors.customerId}
              </p>
            )}
          </div>

          {/* Issue Description */}
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">
              Issue Description <span className="text-zomato-red">*</span>
            </label>
            <textarea
              rows={4}
              className={`input-field resize-none leading-relaxed ${errors.issueDescription ? 'border-red-600 focus:ring-red-600' : ''}`}
              placeholder="Describe the issue with your order — wrong items, missing items, damaged packaging, etc."
              value={formData.issueDescription}
              onChange={(e) => setFormData((p) => ({ ...p, issueDescription: e.target.value }))}
              disabled={isProcessing}
            />
            {errors.issueDescription && (
              <p className="mt-1 text-xs text-red-500 flex items-center gap-1">
                <AlertTriangle size={11} /> {errors.issueDescription}
              </p>
            )}
          </div>
        </div>

        {/* ── Image Upload Section ── */}
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <ImageIcon size={16} className="text-zinc-400" />
            <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-wide">
              Evidence Photos
            </h2>
            <span className="ml-auto text-xs text-zinc-600">Both required</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Bill Zone */}
            <div>
              <DropZone
                label="Drop Bill Photo"
                icon={Camera}
                preview={billPreview}
                dragActive={dragOverBill}
                onDrop={onBillDrop}
                onDragOver={onBillDragOver}
                onDragLeave={onBillDragLeave}
                onFileChange={onBillFileChange}
                onRemove={() => { setBillImage(null); setBillPreview(null); }}
                inputRef={billInputRef}
              />
              {errors.billImage && (
                <p className="mt-1.5 text-xs text-red-500 flex items-center gap-1">
                  <AlertTriangle size={11} /> {errors.billImage}
                </p>
              )}
            </div>

            {/* Food Zone */}
            <div>
              <DropZone
                label="Drop Food Photo"
                icon={Utensils}
                preview={foodPreview}
                dragActive={dragOverFood}
                onDrop={onFoodDrop}
                onDragOver={onFoodDragOver}
                onDragLeave={onFoodDragLeave}
                onFileChange={onFoodFileChange}
                onRemove={() => { setFoodImage(null); setFoodPreview(null); }}
                inputRef={foodInputRef}
              />
              {errors.foodImage && (
                <p className="mt-1.5 text-xs text-red-500 flex items-center gap-1">
                  <AlertTriangle size={11} /> {errors.foodImage}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* ── Live Status Stepper ── */}
        <StatusStepper submissionState={submissionState} />

        {/* ── Error result ── */}
        {submissionState === 'error' && result && (
          <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-4 flex items-start gap-3 animate-[slideUp_0.4s_ease]">
            <AlertTriangle size={18} className="text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-300">Submission Failed</p>
              <p className="text-xs text-red-400/80 mt-0.5">{result.message}</p>
            </div>
          </div>
        )}

        {/* ── Result Panel ── */}
        <ResultPanel result={result} submissionState={submissionState} />

        {/* ── Action Buttons ── */}
        <div className="flex gap-3">
          {canReset ? (
            <button
              type="button"
              onClick={handleReset}
              className="flex-1 btn-primary bg-zinc-700 hover:bg-zinc-600 shadow-none"
            >
              Submit Another Claim
            </button>
          ) : (
            <button
              type="submit"
              disabled={isProcessing}
              className="flex-1 btn-primary"
            >
              {isProcessing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Processing…
                </>
              ) : (
                <>
                  Submit Claim
                  <ChevronRight size={16} />
                </>
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
