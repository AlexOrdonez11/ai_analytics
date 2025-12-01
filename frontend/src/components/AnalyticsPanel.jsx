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
        if (item.type === 'image') {
          return (
            <div
              key={item.id}
              className="rounded-xl border border-neutral-800 p-2 bg-black/40"
            >
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs uppercase tracking-wide text-neutral-400">
                  {item.meta?.type || 'Plot'}
                </span>

                {/* Download button */}
                <a
                  href={item.url}
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs px-2 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200"
                >
                  Download
                </a>
              </div>

              <img
                src={item.url}
                alt={item.meta?.type || 'Analytics plot'}
                className="w-full rounded-lg"
              />
            </div>
          )
        }

        // fallback for other item types if you add them later
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