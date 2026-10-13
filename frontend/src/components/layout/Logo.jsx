// The HemoSmart mark ("refined tile"): a slim ink teardrop on a
// goldfish-yellow tile, with a hairline pulse through it -- blood, read
// by data. Fixed brand colours so the mark looks the same in light and
// dark mode. Kept in sync with public/favicon.svg.
export function LogoMark({ size = 28, title }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className="logo-mark"
         role={title ? "img" : undefined} aria-hidden={title ? undefined : "true"}>
      {title && <title>{title}</title>}
      <rect width="32" height="32" rx="8" fill="#F4C542" />
      <path d="M16 5.2C16 5.2 10 12.6 10 18.8C10 22.3 12.7 25 16 25C19.3 25 22 22.3 22 18.8C22 12.6 16 5.2 16 5.2Z"
            fill="#151413" />
      <path d="M11.8 19.3H13.9L15.2 16.6L17 21.9L18.3 19.3H20.2" fill="none" stroke="#F4C542" strokeWidth="1.3"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Manrope, one ink colour: "Hemo" bold, "Smart" regular.
export function Wordmark() {
  return (
    <span className="brand-name">Hemo<span className="brand-name-accent">Smart</span></span>
  );
}
