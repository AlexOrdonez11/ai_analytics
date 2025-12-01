import React from 'react'
import AnalyticsCard from './AnalyticsCard.jsx'
import Card from './Card.jsx'

export default function AnalyticsPanel({ items, setItems }) {
  if (!items || items.length === 0) {
    return (
      <div className="mb-4 rounded-xl border border-neutral-800 p-3 text-sm text-neutral-400">
        No analytics yet. Ask the analyst for a plot, e.g. “Show the distribution of price”.
      </div>
    )
  }

  return (
    <div className="mb-4 space-y-3">
      {items.map(item => {
        // ---- Plot cards (images) ----
        if (item.type === 'image') {
          const label = item.meta?.type || 'Plot'
          const reportUrl = item.meta?.report_url // forecast tools put this here

          return (
            <div
              key={item.id}
              className="rounded-xl border border-neutral-800 p-2 bg-black/40"
            >
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs uppercase tracking-wide text-neutral-400">
                  {label}
                </span>

                <div className="flex gap-2">
                  {/* Download PNG */}
                  <a
                    href={item.url}
                    download
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs px-2 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200"
                  >
                    Download image
                  </a>

                  {/* Optional: forecast report CSV / doc */}
                  {reportUrl && (
                    <a
                      href={reportUrl}
                      download
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs px-2 py-1 rounded-lg bg-blue-900 hover:bg-blue-800 text-blue-100"
                    >
                      Download forecast CSV
                    </a>
                  )}
                </div>
              </div>

              <img
                src={item.url}
                alt={label}
                className="w-full rounded-lg"
              />
            </div>
          )
        }

        // ---- Optional: pure file/document card (if you add type: 'file' later) ----
        if (item.type === 'file') {
          const label = item.label || item.meta?.model || 'Report'

          return (
            <div
              key={item.id}
              className="rounded-xl border border-neutral-800 p-3 bg-black/40 flex items-center justify-between"
            >
              <div className="flex flex-col">
                <span className="text-xs uppercase tracking-wide text-neutral-400">
                  {label}
                </span>
                {item.meta?.metrics && (
                  <span className="text-[11px] text-neutral-500 mt-1">
                    MAE: {item.meta.metrics.mae?.toFixed?.(2)} · RMSE: {item.meta.metrics.rmse?.toFixed?.(2)}
                  </span>
                )}
              </div>
              <a
                href={item.url}
                download
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-2 py-1 rounded-lg bg-blue-900 hover:bg-blue-800 text-blue-100"
              >
                Download
              </a>
            </div>
          )
        }

        // ---- Fallback for unknown item types ----
        return (
          <div
            key={item.id}
            className="rounded-xl border border-neutral-800 p-3 text-sm text-neutral-200"
          >
            {JSON.stringify(item)}
          </div>
        )
      })}
    </div>
  )
}
