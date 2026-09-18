import { useState } from "react";
import { auth } from "../api";

// Minimal login -- functional, not visually elaborate. Role-based
// dashboards (Day 6) are what actually differentiate the experience
// after sign-in; this screen's only job is getting a token.
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
      setError("Login failed -- check your email and password.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app login-screen">
      <header>
        <h1>HemoSmart</h1>
        <p>Sign in to continue</p>
      </header>
      <form className="card login-card" onSubmit={submit}>
        <label>
          Email
          <input type="email" value={email} required
                 onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Password
          <input type="password" value={password} required
                 onChange={(e) => setPassword(e.target.value)} />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>Sign In</button>
        </div>
        {error && <p className="error">{error}</p>}
      </form>
    </div>
  );
}
