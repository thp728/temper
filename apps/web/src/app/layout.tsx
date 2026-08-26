import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Temper",
    template: "%s · Temper",
  },
  description: "Fine-tuning platform. Dataset in, adapter out.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col bg-neutral-50 text-neutral-900 antialiased">
        <header className="border-b border-neutral-200 bg-white">
          <div className="mx-auto flex w-full max-w-3xl items-center justify-between px-4 py-3">
            <Link
              href="/"
              className="text-lg font-semibold tracking-tight hover:underline"
            >
              Temper
            </Link>
            <nav aria-label="Main">
              <Link
                href="/"
                className="rounded px-2 py-1 text-sm text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900"
              >
                Upload a dataset
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
