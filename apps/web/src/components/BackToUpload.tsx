import Link from "next/link";

// One way back to the start of the journey, wherever a dead end puts you.
export default function BackToUpload() {
  return (
    <Link
      href="/"
      className="rounded-md border border-neutral-300 bg-white px-4 py-2 text-sm font-medium hover:bg-neutral-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
    >
      Back to upload
    </Link>
  );
}
