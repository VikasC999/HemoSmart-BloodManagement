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

// HemoBuddy, the assistant: the brand drop turned into a friendly
// character (gold drop, ink face) on a round ink badge, so it reads as
// "someone to talk to" next to the square HemoSmart tile. Fixed colours.
export function BuddyMark({ size = 32, title }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" className="logo-mark buddy-mark"
         role={title ? "img" : undefined} aria-hidden={title ? undefined : "true"}>
      {title && <title>{title}</title>}
      <circle cx="20" cy="20" r="20" fill="#151413" />
      <path d="M20 6.5C20 6.5 11.5 16.4 11.5 23.6C11.5 28.5 15.3 32.2 20 32.2C24.7 32.2 28.5 28.5 28.5 23.6C28.5 16.4 20 6.5 20 6.5Z"
            fill="#F4C542" />
      <path d="M15.6 18.6C14.9 19.8 14.5 21 14.5 22.1" fill="none" stroke="#FFF3C4" strokeWidth="1.4"
            strokeLinecap="round" />
      <ellipse cx="17" cy="23.4" rx="1.35" ry="1.7" fill="#151413" />
      <ellipse cx="23" cy="23.4" rx="1.35" ry="1.7" fill="#151413" />
      <circle cx="15.3" cy="26.6" r="1.3" fill="#E98B82" opacity="0.85" />
      <circle cx="24.7" cy="26.6" r="1.3" fill="#E98B82" opacity="0.85" />
      <path d="M17.6 27.2C18.9 28.6 21.1 28.6 22.4 27.2" fill="none" stroke="#151413" strokeWidth="1.5"
            strokeLinecap="round" />
    </svg>
  );
}

// Manrope, one ink colour: "Hemo" bold, "Smart" regular.
export function Wordmark() {
  return (
    <span className="brand-name">Hemo<span className="brand-name-accent">Smart</span></span>
  );
}
