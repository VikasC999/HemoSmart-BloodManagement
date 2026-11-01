import { useState } from "react";
import { CircleAlert } from "lucide-react";
import { auth } from "../api";
import { LogoMark, Wordmark } from "./layout/Logo";

// Getting a token is this screen's only job; the role decides the
// workspace you land in.
export default function LoginScreen({ onLoggedIn }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await auth.login(email, password);
      onLoggedIn();
    } catch (err) {
      setError("That email and password don't match an account. Check both and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="login-screen">
      <section className="login-brand-panel" aria-label="HemoSmart">
        <p className="login-mark"><LogoMark size={24} /><Wordmark /></p>
        <div className="login-pitch">
          <h1 className="login-title">Welcome to<br />Hemo<span className="brand-name-accent">Smart</span></h1>
          <p className="login-tagline">Blood management, done right.</p>
        </div>
      </section>
      <section className="login-panel" aria-labelledby="login-heading">
        <h2 id="login-heading" className="login-panel-title">Sign in to your workspace</h2>
        <form className="login-form" onSubmit={submit}>
          <label>
            Work email
            <input type="email" value={email} required autoComplete="username"
                   aria-invalid={error ? "true" : undefined}
                   onChange={(e) => setEmail(e.target.value)} />
          </label>
          <label>
            Password
            <input type="password" value={password} required autoComplete="current-password"
                   aria-invalid={error ? "true" : undefined}
                   onChange={(e) => setPassword(e.target.value)} />
          </label>
          {error && (
            <p className="form-error" role="alert">
              <CircleAlert size={16} aria-hidden="true" /> {error}
            </p>
          )}
          <button type="submit" disabled={loading} className="block">
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}
