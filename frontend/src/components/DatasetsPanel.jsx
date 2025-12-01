import React from 'react'
import Card from './Card.jsx'
import { datasetsApi } from '../lib/api.js'

export default function DatasetsPanel({ projectId }) {
  const [items, setItems] = React.useState([])
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState(null)

  const load = React.useCallback(async () => {
    if (!projectId) return
    setLoading(true); setError(null)
    try {
      const list = await datasetsApi.listProjectDatasets(projectId)
      setItems(list)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [projectId])

  React.useEffect(() => { load() }, [load])

  const hasDataset = items.length > 0

  const onPick = async (e) => {
    if (hasDataset) {         // extra safety
      e.target.value = ''
      return
    }
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    try {
      setLoading(true)
      await datasetsApi.uploadDataset({ project_id: projectId, file })
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  const onDelete = async (id) => {
    try {
      await datasetsApi.deleteDataset(id)
      setItems(prev => prev.filter(x => x.id !== id))
    } catch (e) {
      setError(e.message)
    }
  }

  const onDownload = async (id) => {
    try {
      const url = await datasetsApi.getSignedDownloadUrl(id)
      window.location.href = url
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-neutral-300">
          Datasets ({items.length})
        </h3>
        <label
          className={
            'text-xs px-3 py-1 rounded-xl ' +
            (hasDataset || loading
              ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed opacity-60'
              : 'bg-cyan-600 hover:bg-cyan-500 cursor-pointer')
          }
          title={
            hasDataset
              ? 'A dataset is already loaded. Delete it to upload another.'
              : 'Upload CSV'
          }
        >
          Upload CSV
          <input
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={onPick}
            disabled={hasDataset || loading}
          />
        </label>
      </div>

      {error && <div className="text-sm text-red-400">{error}</div>}
      {loading && <div className="text-sm text-neutral-400">Working…</div>}

      {items.length === 0 ? (
        <Card className="p-4 text-sm text-neutral-400">
          No datasets yet. Upload a CSV.
        </Card>
      ) : (
        <div className="space-y-2">
          {items.map(ds => (
            <Card key={ds.id} className="p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="font-medium text-sm">{ds.name}</div>
                  <div className="text-xs text-neutral-400">
                    {ds.contentType} • {(ds.size / 1024).toFixed(1)} KB •{' '}
                    {new Date(ds.createdAt).toLocaleString?.()}
                  </div>
                  {ds.schema && (
                    <div className="mt-2 text-xs text-neutral-400">
                      Schema:{' '}
                      {Object.entries(ds.schema)
                        .slice(0, 6)
                        .map(([k, v]) => `${k}(${v})`)
                        .join(', ')}
                      {Object.keys(ds.schema).length > 6 ? '…' : ''}
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => onDownload(ds.id)}
                    className="text-xs px-2 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700"
                  >
                    Download
                  </button>
                  <button
                    onClick={() => onDelete(ds.id)}
                    className="text-xs px-2 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {ds.preview?.length > 0 && (
                <div className="mt-3 overflow-x-auto">
                  <table className="text-xs w-full border-collapse">
                    <thead className="text-neutral-400">
                      <tr>
                        {Object.keys(ds.preview[0]).map(h => (
                          <th
                            key={h}
                            className="text-left border-b border-neutral-800 pb-1 pr-4"
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {ds.preview.slice(0, 5).map((row, i) => (
                        <tr key={i}>
                          {Object.keys(row).map(k => (
                            <td
                              key={k}
                              className="py-1 pr-4 border-b border-neutral-900"
                            >
                              {String(row[k])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
