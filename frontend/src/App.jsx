import React from 'react'
import Header from './components/Header.jsx'
import { useAuth } from './hooks/useAuth.js'
import { useHashRoute, routes } from './hooks/useHashRoute.js'
import { useProjectStore } from './hooks/useProjectStore.js'
import LoginPage from './pages/LoginPage.jsx'
import ProjectsHome from './pages/ProjectsHome.jsx'
import Workspace from './pages/Workspace.jsx'

export default function App() {
  const { user, loginEmail, register, logout } = useAuth()
  const { hash, push } = useHashRoute()
  const { projects, createProject } = useProjectStore(user)

  // Start empty; decide later once projects are loaded
  const [currentProjectId, setCurrentProjectId] = React.useState(() => {
    try { return JSON.parse(localStorage.getItem('app:currentProject')) || null } catch { return null }
  })
  React.useEffect(() => {
    if (currentProjectId) {
      localStorage.setItem('app:currentProject', JSON.stringify(currentProjectId))
    }
  }, [currentProjectId])

  // Routing after login
  React.useEffect(() => {
    if (!user) {
      push(routes.login)
      return
    }
    // If no project selected yet, choose the most recent loaded one
    if (!currentProjectId && projects.length > 0) {
      const fallback = projects[0].id // listProjectsByUser is sorted by updatedAt desc in our backend
      setCurrentProjectId(fallback)
    }
    if (hash !== routes.workspace) {
      push(routes.workspace)
    }
  }, [user, projects.length])

  const onOpenProject = (pid) => {
    setCurrentProjectId(pid)
    // persist immediately to avoid any race with effects
    localStorage.setItem('app:currentProject', JSON.stringify(pid))
    push(routes.workspace)
  }

  const currentProject = React.useMemo(
    () => projects.find(p => p.id === currentProjectId),
    [projects, currentProjectId]
  )

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <Header user={user} onLogout={logout} onNav={push} />

      {!user ? (
        <LoginPage
          onLoginEmail={async (email, password) => { await loginEmail(email, password); push(routes.workspace) }}
          onRegister={async (payload) => { await register(payload); push(routes.workspace) }}
        />
      ) : hash === routes.projects ? (
        <ProjectsHome
          projects={projects}
          onCreate={async (name, description='') => {
            const p = await createProject(name, description)
            onOpenProject(p.id)
          }}
          onOpen={onOpenProject}
        />
      ) : (
        <Workspace
          projectId={currentProjectId}
          projectName={currentProject?.name}
        />
      )}  
    </div>
  )
}