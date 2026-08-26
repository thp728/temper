"use client";

import { useEffect, useRef } from "react";

// Spec 007 puts focus management on navigation alongside labelling and
// announcements: a route change that leaves the keyboard where it was is a
// route change a screen reader never announces. The report's heading receives
// focus on mount, so arriving here starts reading at the top.
export default function FocusHeading({
  children,
}: {
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, []);

  return (
    <h1
      ref={ref}
      id="report-heading"
      tabIndex={-1}
      className="text-2xl font-semibold break-all outline-none"
    >
      {children}
    </h1>
  );
}
