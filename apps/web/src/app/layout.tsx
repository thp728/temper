import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import DemoBanner from "@/components/DemoBanner";
import Sidebar from "@/components/Sidebar";
import { cn } from "@/lib/utils";
import { isZeroCostMode } from "@/lib/demo-mode";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

// The demonstration marking is read at request time, never baked at build
// time: the compose image builds once and the reviewer's `docker compose up`
// decides the mode, so a statically-prerendered banner would freeze the wrong
// answer into every page.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: {
    default: "Temper",
    template: "%s - Temper",
  },
  description: "Fine-tuning platform. Dataset in, artifact out.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const zeroCost = isZeroCostMode();
  return (
    <html
      lang="en"
      className={cn("dark font-sans", geist.variable, geistMono.variable)}
      style={{ colorScheme: "dark" }}
    >
      <body className="flex min-h-screen flex-col bg-background text-foreground antialiased">
        {zeroCost ? <DemoBanner /> : null}
        <div className="flex flex-1">
          <Sidebar />
          <main className="w-full min-w-0 flex-1 px-4 py-8 md:px-8">
            <div className="mx-auto w-full max-w-[1200px]">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
