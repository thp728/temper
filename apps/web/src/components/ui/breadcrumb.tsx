import * as React from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

// Primitive sourced from shadcn/ui (apps/web/AGENTS.md: UI primitives come
// from shadcn/ui, added via the CLI, owned in-repo). Extended through
// className rather than forking. Kept deliberately close to the shadcn
// reference so a future `pnpm dlx shadcn add breadcrumb` diff is readable.
// Separator default is the dataset-detail wireframe's "/" (hairline-strong),
// not the chevron that shadcn ships with -- switch back by passing a
// different child to BreadcrumbSeparator.

function Breadcrumb({ className, ...props }: React.ComponentProps<"nav">) {
  return (
    <nav
      aria-label="breadcrumb"
      data-slot="breadcrumb"
      className={cn(className)}
      {...props}
    />
  );
}

function BreadcrumbList({
  className,
  ...props
}: React.ComponentProps<"ol">) {
  return (
    <ol
      data-slot="breadcrumb-list"
      className={cn(
        "flex flex-wrap items-center gap-1.5 break-words text-sm text-muted-foreground sm:gap-2.5",
        className,
      )}
      {...props}
    />
  );
}

function BreadcrumbItem({
  className,
  ...props
}: React.ComponentProps<"li">) {
  return (
    <li
      data-slot="breadcrumb-item"
      className={cn("inline-flex items-center gap-1.5", className)}
      {...props}
    />
  );
}

function BreadcrumbLink({
  asChild,
  className,
  children,
  ...props
}: React.ComponentProps<"a"> & { asChild?: boolean }) {
  // asChild lets a Next.js <Link> inherit the styling. Rather than pulling
  // in @radix-ui/react-slot, clone the single child and merge the classes
  // (the breadcrumb case is always a single <a> child).
  if (asChild && React.isValidElement(children)) {
    const child = children as React.ReactElement<Record<string, unknown>>;
    return React.cloneElement(child as React.ReactElement<unknown>, {
      ...(props as unknown as Record<string, unknown>),
      "data-slot": "breadcrumb-link",
      className: cn(
        "transition-colors hover:text-foreground",
        className,
        (child.props as { className?: string }).className,
      ),
    } as unknown as Record<string, unknown>);
  }
  return (
    <a
      data-slot="breadcrumb-link"
      className={cn("transition-colors hover:text-foreground", className)}
      {...props}
    >
      {children}
    </a>
  );
}

function BreadcrumbPage({
  className,
  ...props
}: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="breadcrumb-page"
      role="link"
      aria-disabled="true"
      aria-current="page"
      className={cn("font-normal text-foreground", className)}
      {...props}
    />
  );
}

function BreadcrumbSeparator({
  children,
  className,
  ...props
}: React.ComponentProps<"li">) {
  return (
    <li
      data-slot="breadcrumb-separator"
      role="presentation"
      aria-hidden="true"
      className={cn("[&>svg]:size-3.5", className)}
      {...props}
    >
      {children ?? <ChevronRight className="size-3.5 opacity-40" />}
    </li>
  );
}

// Matches the wireframe dataset-detail.html:344 pattern -- a "/" in
// hairline-strong, not the shadcn chevron. Use as:
// <BreadcrumbSeparator><span className="text-muted-foreground/40">/</span></BreadcrumbSeparator>
function BreadcrumbSlashSeparator({
  className,
  ...props
}: React.ComponentProps<"li">) {
  return (
    <BreadcrumbSeparator
      className={cn("text-muted-foreground/40", className)}
      {...props}
    >
      <span aria-hidden="true">/</span>
    </BreadcrumbSeparator>
  );
}

function BreadcrumbEllipsis({
  className,
  ...props
}: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="breadcrumb-ellipsis"
      role="presentation"
      aria-hidden="true"
      className={cn("flex size-9 items-center justify-center", className)}
      {...props}
    >
      <span className="size-4">…</span>
      <span className="sr-only">More</span>
    </span>
  );
}

export {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbPage,
  BreadcrumbSeparator,
  BreadcrumbSlashSeparator,
  BreadcrumbEllipsis,
};
