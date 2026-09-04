"use client";

import Link from "next/link";
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsUpDown,
  Inbox,
  ListFilter,
  Network,
  Plus,
  Search,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import EmptyStatePanel from "@/components/EmptyStatePanel";
import StatusPill from "@/components/StatusPill";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { Input } from "@/components/ui/input";
import { formatDuration, formatExactTimestamp, formatTimestamp } from "@/lib/jobs/display";
import type { JobRecord } from "@/lib/api/generated/client";

// The jobs list (#14's screen, restyled): every job, each with its outcome
// beside it and a link to its full record -- the event history, and the
// artifact when there is one. The table is the dashboard's finished table
// (`DashboardView`): the same rounded card shell, header treatment, row
// rhythm and right-aligned numerics, so one table has one look on both
// screens. Columns follow `docs/wireframes/jobs.html` (job, status, model,
// dataset, created, duration) but only what the API publishes: duration reads
// the frozen `actuals.duration_s` and shows "—" while a job is still active,
// because a live elapsed clock is the dashboard's active table's business.
// Sorting and the status/model filters are client-side over the fetched list
// (the datasets screen's search precedent): the API has no sort/filter
// params, so no control here implies a server query. The status renders through the same
// pill the dashboard uses, so one status has one look on both screens; the
// stable error code is the job record's detail, not repeated in the list.
//
// Intent:     an engineer between launches, scanning history to find a job
//             and jump to its record; quiet console, same as the dashboard.
// Hierarchy:  the job-id link leads (primary), the status pill seconds,
//             model/dataset/created/duration demote to secondary text; the
//             sort control lives in the header it orders, icon-only.
// Palette:    existing tokens only -- no new hues.
// Depth:      borders-only, rounded-[12px], no shadows.
// Surfaces:   bg-card shell, rows bg-background/50.
// Typography: th medium/muted; model in code; created/duration tabular-nums.
// Spacing:    4px base; px-4 py-3 head, px-4 py-4 cells.

// The way a text link reads: the accent for emphasis, a step lighter on
// hover, never an underline.
const LINK = "text-primary transition-colors hover:text-primary-hover";

const PAGE_SIZE = 10;

type SortKey = "id" | "status" | "base_model" | "dataset" | "created" | "duration";
type SortDir = "asc" | "desc";

// The direction a column starts in: time-like columns lead with the largest,
// text columns read A-Z.
const SORT_DEFAULT_DIR: Record<SortKey, SortDir> = {
  id: "asc",
  status: "asc",
  base_model: "asc",
  dataset: "asc",
  created: "desc",
  duration: "desc",
};

// A single-select filter over the fetched list (`docs/wireframes/jobs.html`'s
// "Status: All" / "Model: Any"): a Radix menu, not a native select, so the
// trigger reads as the wireframe's icon-label-chevron button. The chevron
// stays centered because every element in the trigger shares one inline-flex
// row -- icon, label and chevron are all `size-4` siblings under
// `items-center`, never a floated or absolutely positioned afterthought.
function FilterMenu({
  name,
  icon,
  prefix,
  current,
  allValue,
  allLabel,
  options,
  onChange,
}: {
  // The accessible name, stable while the visible value changes.
  name: string;
  icon: ReactNode;
  // The dimmed prefix in the trigger ("Status", "Base model").
  prefix: string;
  current: string;
  allValue: string;
  allLabel: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  const selected = options.find((o) => o.value === current);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" variant="outline" aria-label={name}>
          <span aria-hidden="true" className="inline-flex [&_svg]:size-4">
            {icon}
          </span>
          <span>
            <span className="text-muted-foreground">{prefix}:</span>{" "}
            {selected ? selected.label : allLabel}
          </span>
          <ChevronDown aria-hidden="true" className="size-4 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {[{ value: allValue, label: allLabel }, ...options].map((option) => (
          <DropdownMenuItem
            key={option.value}
            onSelect={() => onChange(option.value)}
          >
            <span aria-hidden="true" className="inline-flex w-4">
              {current === option.value && <Check className="size-4" />}
            </span>
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onSort,
  align = "left",
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: SortDir;
  onSort: (key: SortKey) => void;
  align?: "left" | "right";
}) {
  const active = sortKey === activeKey;
  return (
    <th
      scope="col"
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
      className={`px-4 py-3 font-medium text-muted-foreground${align === "right" ? " text-right" : ""}`}
    >
      {/* No aria-label: the button's name is the column title itself, so the
          header keeps the name the journeys assert on. */}
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 font-medium transition-colors hover:text-foreground${active ? " text-foreground" : ""}${align === "right" ? " flex-row-reverse" : ""}`}
      >
        {label}
        <span aria-hidden="true" className="inline-flex">
          {active ? (
            dir === "asc" ? (
              <ArrowUp className="size-3.5" />
            ) : (
              <ArrowDown className="size-3.5" />
            )
          ) : (
            <ChevronsUpDown className="size-3.5 opacity-50" />
          )}
        </span>
      </button>
    </th>
  );
}

export default function JobsView({
  jobs,
  datasetNames,
  now,
}: {
  jobs: JobRecord[];
  datasetNames: Record<string, string>;
  // The server page's clock reading, not the client's own: the relative
  // dates below render on the server first, and hydration must see the same
  // strings the server sent (`RunningJobView`'s rule).
  now: number;
}) {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("any");
  const [sortKey, setSortKey] = useState<SortKey>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(0);

  const statuses = useMemo(
    () => Array.from(new Set(jobs.map((job) => job.status))).sort(),
    [jobs],
  );

  const models = useMemo(
    () => Array.from(new Set(jobs.map((job) => job.base_model))).sort(),
    [jobs],
  );

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(SORT_DEFAULT_DIR[key]);
    }
  }

  const visibleJobs = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = jobs.filter((job) => {
      if (statusFilter !== "all" && job.status !== statusFilter) return false;
      if (modelFilter !== "any" && job.base_model !== modelFilter)
        return false;
      if (!q) return true;
      const dataset = datasetNames[job.dataset_id] ?? job.dataset_id;
      return (
        job.id.toLowerCase().includes(q) ||
        job.base_model.toLowerCase().includes(q) ||
        job.status.toLowerCase().includes(q) ||
        dataset.toLowerCase().includes(q)
      );
    });
    const dirSign = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      switch (sortKey) {
        case "created":
          return dirSign * (a.created_at - b.created_at);
        case "duration": {
          // A job with no frozen duration has nothing to order by: it sinks
          // below every measured one in either direction.
          const da = a.actuals?.duration_s;
          const db = b.actuals?.duration_s;
          if (da == null && db == null) return 0;
          if (da == null) return 1;
          if (db == null) return -1;
          return dirSign * (da - db);
        }
        case "dataset": {
          const da = datasetNames[a.dataset_id] ?? a.dataset_id;
          const db = datasetNames[b.dataset_id] ?? b.dataset_id;
          return dirSign * da.localeCompare(db);
        }
        case "id":
          return dirSign * a.id.localeCompare(b.id);
        case "status":
          return dirSign * a.status.localeCompare(b.status);
        case "base_model":
          return dirSign * a.base_model.localeCompare(b.base_model);
      }
    });
  }, [jobs, datasetNames, query, statusFilter, modelFilter, sortKey, sortDir]);

  // Pages over the filtered, sorted list, ten rows at a time. The page
  // clamps rather than resets when the list shrinks beneath it, so narrowing
  // a search never lands on an empty page; starting a new search or filter
  // goes back to the first page below.
  const totalPages = Math.max(1, Math.ceil(visibleJobs.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, visibleJobs.length);
  const pageJobs = visibleJobs.slice(start, end);

  return (
    <section aria-labelledby="jobs-heading" className="space-y-8">
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbPage>Jobs</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1">
          <h1
            id="jobs-heading"
            className="text-2xl font-semibold tracking-tight text-balance"
          >
            Manage and monitor your fine-tuning jobs
          </h1>
          <p className="text-sm text-muted-foreground">
            Every job, newest first. Each links to its full record: the event
            history, and the artifact when there is one.
          </p>
        </div>
        {/* A job starts with a dataset: the launch screen needs one, so the
            action lands on the dataset picker rather than a page that would
            only refuse with `no_dataset`. */}
        <Button asChild>
          <Link href="/datasets">
            <Plus aria-hidden="true" className="size-4" />
            Start a new job
          </Link>
        </Button>
      </div>

      {jobs.length === 0 ? (
        <EmptyStatePanel
          icon={Inbox}
          heading="No jobs yet"
          headingAs="h2"
          action={
            <Button asChild>
              <Link href="/datasets">Select a dataset</Link>
            </Button>
          }
        >
          Every job you launch appears here, newest first, with its status
          beside it.
        </EmptyStatePanel>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative w-full sm:w-64">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                type="search"
                aria-label="Search jobs"
                placeholder="Search by job, model, dataset…"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setPage(0);
                }}
                className="pl-8"
              />
            </div>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              <FilterMenu
                name="Filter by status"
              icon={<ListFilter />}
              prefix="Status"
              current={statusFilter}
              allValue="all"
              allLabel="All"
              options={statuses.map((status) => ({
                value: status,
                label: status,
              }))}
              onChange={(value) => {
                setStatusFilter(value);
                setPage(0);
              }}
            />
            <FilterMenu
              name="Filter by base model"
              icon={<Network />}
              prefix="Base model"
              current={modelFilter}
              allValue="any"
              allLabel="Any"
              options={models.map((model) => ({
                value: model,
                label: model,
              }))}
              onChange={(value) => {
                setModelFilter(value);
                setPage(0);
              }}
            />
            </div>
          </div>
          {visibleJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No jobs match the current search and filters.{" "}
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setStatusFilter("all");
                  setModelFilter("any");
                  setPage(0);
                }}
                className={LINK}
              >
                Clear search and filters
              </button>
            </p>
          ) : (
            <div className="overflow-hidden rounded-[12px] border bg-card">
              <div className="overflow-x-auto">
                <table
                  aria-label="Jobs"
                  className="w-full table-fixed border-collapse text-sm tabular-nums"
                >
                  {/* No per-column widths: table-fixed splits the six columns
                      evenly. */}
                  <thead>
                    <tr className="border-b text-left">
                      <SortHeader
                        label="Job"
                        sortKey="id"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                      <SortHeader
                        label="Status"
                        sortKey="status"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                      <SortHeader
                        label="Base model"
                        sortKey="base_model"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                      <SortHeader
                        label="Dataset"
                        sortKey="dataset"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                      <SortHeader
                        label="Created"
                        sortKey="created"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                      <SortHeader
                        label="Duration"
                        sortKey="duration"
                        activeKey={sortKey}
                        dir={sortDir}
                        onSort={handleSort}
                        align="right"
                      />
                    </tr>
                  </thead>
                  <tbody>
                    {pageJobs.map((job) => {
                      const dataset =
                        datasetNames[job.dataset_id] ?? job.dataset_id;
                      const duration = job.actuals?.duration_s;
                      return (
                        <tr
                          key={job.id}
                          className="border-b border-border bg-background/50 last:border-b-0"
                        >
                          <td className="truncate px-4 py-4">
                            <Link
                              href={`/jobs/${job.id}`}
                              className={LINK}
                              title={job.id}
                            >
                              {job.id}
                            </Link>
                          </td>
                          <td className="px-4 py-4">
                            <StatusPill status={job.status} />
                          </td>
                          <td className="truncate px-4 py-4">
                            {/* The model id is what every other surface calls it. */}
                            <code title={job.base_model}>
                              {job.base_model}
                            </code>
                          </td>
                          <td
                            className="truncate px-4 py-4 text-muted-foreground"
                            title={dataset}
                          >
                            {dataset}
                          </td>
                          <td
                            className="truncate px-4 py-4 text-muted-foreground"
                            title={formatExactTimestamp(job.created_at)}
                          >
                            {formatTimestamp(job.created_at, now)}
                          </td>
                          <td className="px-4 py-4 text-right">
                            {duration != null
                              ? formatDuration(duration)
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {visibleJobs.length > PAGE_SIZE && (
            <div className="flex items-center justify-between px-1 text-sm">
              <span
                className="text-muted-foreground"
                aria-live="polite"
                data-testid="jobs-pagination-info"
              >
                Showing {start + 1}–{end} of {visibleJobs.length} · Page{" "}
                {safePage + 1} of {totalPages}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="icon-sm"
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={safePage === 0}
                  aria-label="Previous page"
                >
                  <ChevronLeft aria-hidden="true" />
                </Button>
                <Button
                  variant="outline"
                  size="icon-sm"
                  onClick={() =>
                    setPage((p) => Math.min(totalPages - 1, p + 1))
                  }
                  disabled={safePage >= totalPages - 1}
                  aria-label="Next page"
                >
                  <ChevronRight aria-hidden="true" />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
