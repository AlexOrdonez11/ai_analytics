import React from 'react'

const LONG_THRESHOLD = 700 // chars before we collapse

export default function ChatBubble({ role, text }) {
  const isUser = role === 'user'
  const [expanded, setExpanded] = React.useState(false)

  const safeText = text ?? ''
  const isLong = safeText.length > LONG_THRESHOLD
  const displayText =
    !isLong || expanded
      ? safeText
      : safeText.slice(0, LONG_THRESHOLD) + '…'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`
          max-w-[52rem] md:max-w-[40rem]
          rounded-2xl px-4 py-3
          text-sm leading-7 whitespace-pre-wrap break-words
          ${isUser
            ? 'bg-cyan-600 text-white'
            : 'bg-neutral-900 text-neutral-50'}
        `}
      >
        {displayText}

        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 text-xs text-neutral-200 underline underline-offset-2"
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
    </div>
  )
}