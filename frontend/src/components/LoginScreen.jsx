import { useState } from "react";
import { CircleAlert, LockKeyhole } from "lucide-react";
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
    <div className="login-screen">
      <main className="login-panel">
        <div className="login-brand">
          <LogoMark size={44} />
          <Wordmark />
        </div>
        <h1>Sign in to your workspace</h1>
        <p className="login-intro">
          Forecast blood demand, predict transfusion needs and keep stock from expiring unused.
        </p>
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
        <p className="login-foot">
          <LockKeyhole size={13} aria-hidden="true" />
          Your role decides what you see. Every action is recorded in the audit log.
        </p>
      </main>
    </div>
  );
}
