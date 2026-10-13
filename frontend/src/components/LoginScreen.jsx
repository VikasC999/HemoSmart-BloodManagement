import { useState } from "react";
import { CircleAlert } from "lucide-react";
import { auth } from "../api";
import { BrandMark } from "./layout/AppShell";

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
      <div className="login-panel">
        <BrandMark size={28} />
        <h1>Sign in</h1>
        <p className="login-intro">
          Blood demand prediction, inventory and transfusion support for your hospital.
        </p>
        <form className="login-form" onSubmit={submit} noValidate={false}>
          <label>
            Email
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
      </div>
    </div>
  );
}
