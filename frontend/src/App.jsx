import { useState } from 'react';
import CustomerClaimForm from './components/CustomerClaimForm.jsx';
import AdminDashboard from './components/AdminDashboard.jsx';

export default function App() {
  const [activeTab, setActiveTab] = useState('claim');

  return (
    <div className="min-h-screen bg-zinc-950 text-white flex flex-col">
      {/* ── Navbar ── */}
      <header className="sticky top-0 z-50 border-b border-zinc-800/60 bg-zinc-950/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3 select-none">
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center text-lg shadow-lg"
                style={{ background: '#E23744' }}
                aria-label="ZomatoRefund Logo"
              >
                🍴
              </div>
              <div className="flex flex-col leading-tight">
                <span className="font-bold text-base tracking-tight text-white">
                  ZomatoRefund
                </span>
                <span className="text-[10px] text-zinc-500 font-medium tracking-widest uppercase">
                  Order Dispute Portal
                </span>
              </div>
            </div>

            {/* Tab Switcher */}
            <nav className="flex items-center gap-1 bg-zinc-900 rounded-xl p-1">
              <button
                onClick={() => setActiveTab('claim')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  activeTab === 'claim'
                    ? 'bg-zomato-red text-white shadow-md'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                Submit Claim
              </button>
              <button
                onClick={() => setActiveTab('admin')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  activeTab === 'admin'
                    ? 'bg-zomato-red text-white shadow-md'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                Admin Dashboard
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 overflow-y-auto">
        {activeTab === 'claim' ? <CustomerClaimForm /> : <AdminDashboard />}
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-zinc-800/60 py-4 text-center text-xs text-zinc-600">
        © 2026 ZomatoRefund · Dispute Resolution Engine v2.0
      </footer>
    </div>
  );
}
