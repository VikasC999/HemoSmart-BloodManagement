import { Component, useCallback, useEffect, useState } from "react";
import { auth } from "./api";
import AppShell, { PageHeader } from "./components/layout/AppShell";
import LoginScreen from "./components/LoginScreen";
import { ErrorState, Skeleton } from "./components/ui";
import { NAVIGATION, sectionsFor } from "./navigation";

// One page crashing must not blank the whole app: show where it failed,
// keep the navigation usable. Keyed by section, so moving away resets it.
class SectionBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error("Section crashed:", error);
  }

  render() {
    if (this.state.error) {
      return (
        <ErrorState what="This page hit an error and couldn't be shown." error={this.state.error}
                    stillValid="The rest of HemoSmart still works; pick another page or reload." />
      );
    }
    return this.props.children;
  }
}

// The current section lives in the URL hash (#inventory), so a refresh
// or the Back button keeps you where you were -- without a router.
function useHash() {
  const [hash, setHash] = useState(() => window.location.hash.slice(1));
  useEffect(() => {
    const onChange = () => setHash(window.location.hash.slice(1));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

function WorkspaceSkeleton() {
  return (
    <div className="page" role="status" aria-label="Loading your workspace">
      <div className="page-header">
        <div style={{ display: "grid", gap: 10, width: "min(420px, 100%)" }}>
          <Skeleton height={30} width="60%" />
          <Skeleton height={16} />
        </div>
      </div>
      <Skeleton height={220} style={{ borderRadius: "var(--radius-lg)" }} />
    </div>
  );
}

export default function App() {
  const [loggedIn, setLoggedIn] = useState(auth.isLoggedIn());
  const [user, setUser] = useState(null);
  const [userError, setUserError] = useState(null);
  const hash = useHash();

  const loadUser = useCallback(() => {
    setUserError(null);
    auth.me().then(setUser).catch(setUserError);
  }, []);

  useEffect(() => {
    if (loggedIn) loadUser();
  }, [loggedIn, loadUser]);

  if (!loggedIn) {
    return <LoginScreen onLoggedIn={() => setLoggedIn(true)} />;
  }

  const logout = () => {
    auth.logout();
    setLoggedIn(false);
    setUser(null);
    window.location.hash = "";
  };

  const groups = user ? NAVIGATION[user.role] || [] : [];
  const sections = user ? sectionsFor(user.role) : [];
  const active = sections.find((s) => s.id === hash) || sections[0];

  let content;
  if (userError) {
    content = (
      <div className="page">
        <ErrorState what="Couldn't load your account." error={userError} onRetry={loadUser} />
      </div>
    );
  } else if (!user) {
    content = <WorkspaceSkeleton />;
  } else if (!active) {
    content = (
      <div className="page">
        <ErrorState what={`No workspace is set up for the role "${user.role}".`}
                    error="Ask an administrator to check your account's role." />
      </div>
    );
  } else {
    content = (
      <div className={`page layout-${active.layout}`} key={active.id}>
        {active.ownHeader ? <SectionBoundary>{active.render(active.props)}</SectionBoundary> : (
          <>
            <PageHeader title={active.title} description={active.description} />
            <div className="page-body"><SectionBoundary>{active.render(active.props)}</SectionBoundary></div>
          </>
        )}
      </div>
    );
  }

  return (
    <AppShell user={user} groups={groups} activeId={active?.id} onLogout={logout}>
      {content}
    </AppShell>
  );
}
