// The brand mark: a hollow delta, the product's logo. A LoRA adapter is a
// weight delta -- ΔW -- applied to a base model whose weights are never
// modified, so the artifact this product ships is literally a difference; it
// also reads as the predicted-vs-actual delta the calibration surface
// reports.
//
// Path from Boxicons `delta-filled`, MIT licensed (https://boxicons.com).
// This is the one deliberate exception to "every icon comes from
// lucide-react": it is a single asset, not in Lucide, and a second icon
// library for one path would be silly. The caller sets size and colour with
// `className`; the mark inherits `currentColor` and hardcodes none. Wherever
// a "Temper" wordmark sits beside it the mark is decorative, hence
// `aria-hidden` lives here: the accessible name comes from the text, never
// the mark.
export default function TemperMark({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
      className={className}
    >
      <path d="m3,21h18c.35,0,.68-.18.86-.48.18-.3.19-.67.03-.98L12.88,2.53c-.35-.66-1.42-.66-1.77,0L2.12,19.53c-.16.31-.15.68.03.98.18.3.51.48.86.48Zm7.99-13.95l6.32,11.95H4.66l6.32-11.95Z" />
    </svg>
  );
}
