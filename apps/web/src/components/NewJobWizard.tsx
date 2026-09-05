"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  CircleHelp,
  Cpu,
  Database,
  Hash,
  Info,
  ListChecks,
  Loader2,
  Network,
  Pencil,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import AdmitModelForm from "@/components/AdmitModelForm";
import AdvancedSurface from "@/components/AdvancedSurface";
import ImportForm from "@/components/ImportForm";
import ProbeResultView from "@/components/ProbeResultView";
import ReportView from "@/components/ReportView";
import UploadForm from "@/components/UploadForm";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import QuoteView, { type OverrideRefusal } from "@/components/QuoteView";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  createJobV1JobsPost,
  getDatasetV1DatasetsDatasetIdGet,
  getJobSpecPreviewV1JobsSpecGet,
  listJobsV1JobsGet,
  getQuoteV1QuotesGet,
  recomputeQuoteV1QuotesPost,
  type AdvancedSurface as AdvancedSurfaceModel,
  type AdmittedModel,
  type DatasetAccepted,
  type DatasetRecord,
  type DecisionOverride,
  type JobRecord,
  type JobSpecPreview,
  type ModelCatalog,
  type Quote,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";
import {
  formatFriendlyDurationRange,
  formatMinorCostRange,
  formatTimestamp,
} from "@/lib/jobs/display";
import { cn } from "@/lib/utils";

// Intent:     an ML practitioner starting a fine-tuning job from nothing —
//             no dataset pre-picked, no model pre-picked — who must reach
//             launch with no cost surprise. Feels like a lab ledger: dense,
//             precise, mono numbers, whisper borders.
// Hierarchy:  one focal point per step wins — Step 1 the picked cards
//             (dataset radio, then the selected model card via border-primary
//             + ring), Step 2 the editable smart defaults, Step 3 the quote
//             hero (cost/duration in mono tabular), Step 4 the Launch action +
//             frozen-spec receipt. The estimate rail repeats the focal numbers
//             beside every step at a lower tier, never competing with them —
//             its sections check off as their step completes.
// Palette:    DESIGN.md Linear tokens — canvas #010102, card #09090b, border
//             #23252a, primary #5e6ad2 spent only on action/focus/selection,
//             amber for money/commit marks, success #27a644. Money reads
//             amber, action reads lavender.
// Depth:      borders-only + surface ladder (canvas → card → popover). No
//             shadows on dark — matches DESIGN.md, fits a dense tool.
// Surfaces:   same hue, lightness steps only; sidebar same as canvas.
// Typography: Geist Sans + Geist Mono (layout). Eyebrow 11/500/tracked/muted
//             for step kicker, section titles 20/600/tight, body 14/400, mono
//             tabular for every number. Weight + color do hierarchy, not size
//             alone.
// Spacing:    base 4px, workbench-tight: card p-4/5, section gap-6, content
//             max 1200px with a 2:1 main-to-rail split on step 1. A tool
//             panel, not a brochure.
//
// The launch surface as a four-step wizard — Model & dataset, then
// Hyperparameters, then Compute & hardware, then Review & launch — with the
// estimate rail beside step 1. The state machine is LaunchForm's — dataset,
// selected model, quote, overrides, hyperparameters, delivery — only the
// rendering is stepped. /jobs/new always shows the wizard, even with no
// dataset picked yet: the dataset is chosen on step 1 and its preview
// re-requested from the server, so leaving the screen is never required and
// the preview can never disagree with what POST /v1/jobs freezes. Every
// accessible name LaunchForm exposed is kept verbatim (radios by repo, "The
// job specification", "Cost and time estimate", "Why this configuration",
// "Advanced settings", "Launch job"), so the journeys keep finding controls
// by name, not class. There is no draft: nothing persists until launch, so
// no control promises to keep one.

// Which predictor decisions sit beside the hyperparameters, and which beside
// the compute choice: the names temper_core.decisions records. Training-shape
// choices change what the run does (method and precision beside the adapter
// card, sequence length beside the sequence card); where-it-runs choices
// change what it costs. Anything the predictor adds later renders on neither
// step until it is classified here — a hole the completeness test below
// would rather catch loudly than hide silently.
const COMPUTE_DECISIONS = ["hardware", "device count", "disk"];

type Step = 1 | 2 | 3 | 4;

const STEPS: { n: Step; id: string; name: string; hint: string }[] = [
  { n: 1, id: "sources", name: "Model & Dataset", hint: "Base architecture & corpus" },
  { n: 2, id: "hyperparameters", name: "Hyperparameters", hint: "Smart defaults, editable" },
  { n: 3, id: "compute", name: "Compute & Hardware", hint: "GPU, cost & time" },
  { n: 4, id: "review", name: "Review & Launch", hint: "Pre-flight checks & launch" },
];

function Stepper({
  step,
  onGo,
}: {
  step: Step;
  onGo: (n: Step) => void;
}) {
  return (
    <ol
      aria-label="New job progress"
      className="grid grid-cols-2 gap-2 sm:grid-cols-4"
    >
      {STEPS.map((s) => {
        const done = s.n < step;
        const active = s.n === step;
        return (
          <li key={s.id}>
            <button
              type="button"
              disabled={!done}
              onClick={() => onGo(s.n)}
              aria-current={active ? "step" : undefined}
              className={cn(
                "flex w-full cursor-pointer items-center gap-3 rounded-lg border p-3 text-left transition-colors",
                active
                  ? "border-primary/60 bg-card"
                  : done
                    ? "border-border bg-card hover:border-ring/50"
                    : "border-border bg-card/50 opacity-70",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "flex size-7 shrink-0 items-center justify-center rounded-full font-mono text-xs font-semibold tabular-nums",
                  active
                    ? "bg-primary text-primary-foreground"
                    : done
                      ? "bg-muted text-foreground"
                      : "bg-muted text-muted-foreground",
                )}
              >
                {done ? "✓" : s.n}
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium leading-tight">
                  {s.name}
                </span>
                <span className="block truncate text-xs text-muted-foreground">
                  {s.hint}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

export default function NewJobWizard({
  catalog,
  preview: initialPreview,
  surface,
  admitted = [],
  datasets = [],
}: {
  catalog: ModelCatalog;
  preview: JobSpecPreview | null;
  surface: AdvancedSurfaceModel | null;
  admitted?: AdmittedModel[];
  datasets?: DatasetRecord[];
}) {
  const router = useRouter();
  const [step, setStep] = useState<Step>(1);
  const headingRef = useRef<HTMLHeadingElement>(null);
  // The dataset is picked on the Sources step itself: with no dataset yet
  // there is no preview, and picking one re-requests it from the server, so
  // the spec, the quote and the launch always describe the dataset shown —
  // never a stale one, and never a screen left behind.
  const [datasetId, setDatasetId] = useState<string | null>(
    initialPreview?.dataset.id ?? null,
  );
  const [preview, setPreview] = useState<JobSpecPreview | null>(
    initialPreview,
  );
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewRefusal, setPreviewRefusal] = useState<ApiError | null>(null);
  const lastGoodDataset = useRef<string | null>(
    initialPreview?.dataset.id ?? null,
  );
  const [selected, setSelected] = useState(catalog.default);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [overrides, setOverrides] = useState<DecisionOverride[]>([]);
  const [admittedModels, setAdmittedModels] =
    useState<AdmittedModel[]>(admitted);
  const [hyperparameters, setHyperparameters] = useState<
    Record<string, string>
  >({});
  const [planRefusal, setPlanRefusal] = useState<OverrideRefusal | null>(null);
  const [delivery, setDelivery] = useState<string[]>([]);
  const [modelSource, setModelSource] = useState<"registry" | "import">(
    "registry",
  );
  const [datasetQuery, setDatasetQuery] = useState("");
  // Adding a dataset without leaving the screen: the toggle below swaps
  // the existing list for the same upload/import forms the Datasets page
  // renders, and a fresh arrival joins this local list rather than the
  // server-rendered prop.
  const [datasetSource, setDatasetSource] = useState<"existing" | "add">(
    "existing",
  );
  const [localDatasets, setLocalDatasets] =
    useState<DatasetRecord[]>(datasets);
  const [pendingDataset, setPendingDataset] = useState<{
    id: string;
    filename: string;
  } | null>(null);
  const [pendingNote, setPendingNote] = useState<string | null>(null);
  // The validation report opened in place: fetched when the modal opens, so
  // reading it never navigates away from the wizard.
  const [reportOpen, setReportOpen] = useState(false);
  const [reportRecord, setReportRecord] = useState<DatasetRecord | null>(null);
  const [reportJobs, setReportJobs] = useState<JobRecord[]>([]);
  const [reportJobsError, setReportJobsError] = useState<ApiError | null>(
    null,
  );
  const [reportError, setReportError] = useState<ApiError | null>(null);
  const deliveryOptions = (preview?.delivery_formats ?? [])
    .filter((d) => d.id !== "adapter")
    .map((d) => ({
      id: d.id,
      name: d.id === "merged" ? "Merged model" : "Quantised local format",
      whatFor: d.what_for,
    }));
  const [quoteLoading, setQuoteLoading] = useState(initialPreview !== null);
  const lastGood = useRef<{
    overrides: DecisionOverride[];
    hyperparameters: Record<string, string>;
  }>({ overrides: [], hyperparameters: {} });
  // The predictor's true default per decision: each arriving quote snapshots
  // the choices it did NOT override. After a pin the recomputed quote's
  // `chosen` is the pinned value, so without this the original default would
  // become unselectable -- reselecting it must unpin, not re-pin. State,
  // not a ref: it is read during render and set alongside the quote.
  const [baseChoices, setBaseChoices] = useState<Record<string, string>>({});

  const previewDatasetId = preview?.dataset.id;

  useEffect(() => {
    // No dataset yet means no estimate yet; picking one sets loading
    // explicitly, so this effect never needs to reset state itself.
    if (!preview || !previewDatasetId) return;
    let cancelled = false;
    const hasDemands =
      overrides.length > 0 || Object.keys(hyperparameters).length > 0;
    const request = hasDemands
      ? recomputeQuoteV1QuotesPost({
          dataset_id: previewDatasetId,
          base_model: selected,
          overrides,
          hyperparameters,
        })
      : getQuoteV1QuotesGet({
          dataset_id: previewDatasetId,
          base_model: selected,
        });
    request
      .then((q) => {
        if (cancelled) return;
        setQuote(q);
        lastGood.current = { overrides, hyperparameters };
        setBaseChoices((prev) => {
          const next = { ...prev };
          for (const dec of q?.decisions ?? []) {
            if (!dec.overridden) next[dec.decision] = dec.chosen;
          }
          return next;
        });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 400) {
          setOverrides(lastGood.current.overrides);
          setHyperparameters(lastGood.current.hyperparameters);
          setPlanRefusal({ code: err.code, message: err.message });
        } else {
          setQuote(null);
        }
      })
      .finally(() => {
        if (!cancelled) setQuoteLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected, preview, previewDatasetId, overrides, hyperparameters]);

  function onModelChange(modelId: string) {
    setSelected(modelId);
    setQuote(null);
    setQuoteLoading(true);
    setOverrides([]);
    setHyperparameters({});
    setPlanRefusal(null);
    lastGood.current = { overrides: [], hyperparameters: {} };
  }

  function onAdmitted(record: AdmittedModel) {
    setAdmittedModels((prev) =>
      prev.some((a) => a.id === record.id) ? prev : [...prev, record],
    );
    if (record.probe.ok) {
      onModelChange(record.id);
    }
  }

  // Switching datasets re-requests the preview rather than mutating it
  // locally — the same rule plan overrides follow — and resets the tune
  // state pinned against the previous dataset. A refused dataset reverts to
  // the last good one with its stable code beside the control, so the plan
  // below always describes a configuration that can launch.
  // Stable across renders for the validation poll below, which re-runs only
  // when the pending arrival changes rather than on every keystroke above.
  const onDatasetChange = useCallback(
    (nextId: string) => {
      if (nextId === datasetId || previewLoading) return;
    setDatasetId(nextId);
    setPreviewLoading(true);
    setPreviewRefusal(null);
    setQuote(null);
    setQuoteLoading(true);
    void getJobSpecPreviewV1JobsSpecGet({ dataset_id: nextId })
      .then((next) => {
        setPreview(next);
        lastGoodDataset.current = nextId;
        setOverrides([]);
        setHyperparameters({});
        setPlanRefusal(null);
        lastGood.current = { overrides: [], hyperparameters: {} };
        // The URL follows the pick without navigating: a refresh lands on
        // the same dataset through the server-rendered path.
        try {
          window.history.replaceState(
            null,
            "",
            `/jobs/new?dataset_id=${encodeURIComponent(nextId)}`,
          );
        } catch {
          // A URL that cannot be replaced is cosmetic — the state is already
          // correct, and the launch reads state, never the address bar.
        }
      })
      .catch((err) => {
        setDatasetId(lastGoodDataset.current);
        setPreviewRefusal(
          err instanceof ApiError ? err : NETWORK_ERROR,
        );
      })
      .finally(() => {
        setPreviewLoading(false);
      });
    },
    [datasetId, previewLoading],
  );

  function go(n: Step) {
    setStep(n);
  }

  // A dataset added through the modal: close it and watch validation land,
  // then select the arrival. The accept response carries no report — the
  // record is polled below — so there is no immediate path, only the patient
  // one. Either way the screen never changes.
  function handleNewDataset(record: DatasetAccepted) {
    setDatasetSource("existing");
    setPendingNote(null);
    setPendingDataset({ id: record.id, filename: record.filename });
  }

  // While a fresh arrival validates, poll its record until the report lands
  // and select it then: the first check runs immediately, so an already-done
  // validation selects without waiting out a tick. A dataset that stops
  // validating without a report is left to its report page rather than
  // retried blindly here.
  useEffect(() => {
    if (!pendingDataset) return;
    const pending = pendingDataset;
    let cancelled = false;
    let attempts = 0;
    let timer: ReturnType<typeof setInterval>;
    function finish() {
      clearInterval(timer);
      if (!cancelled) setPendingDataset(null);
    }
    function check() {
      attempts += 1;
      void getDatasetV1DatasetsDatasetIdGet(pending.id)
        .then((record) => {
          if (cancelled) return;
          if (record.report) {
            finish();
            setLocalDatasets((prev) =>
              prev.some((d) => d.id === record.id) ? prev : [...prev, record],
            );
            setPendingNote(null);
            onDatasetChange(record.id);
          } else if (
            record.status !== "importing" &&
            record.status !== "validating"
          ) {
            finish();
            setPendingNote(
              `${pending.filename} stopped validating with no report — open Datasets to see why.`,
            );
          } else if (attempts >= 90) {
            finish();
            setPendingNote(
              `${pending.filename} is still validating — it joins the list above when its report lands.`,
            );
          }
        })
        .catch(() => {
          if (cancelled) return;
          finish();
          setPendingNote(
            `${pending.filename} could not be checked — open Datasets to see why.`,
          );
        });
    }
    check();
    timer = setInterval(check, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [pendingDataset, onDatasetChange]);

  useEffect(() => {
    headingRef.current?.focus();
  }, [step]);

  // The report modal reads the same record and job list the report page
  // server-renders: one fetch for the dataset, one for every job filtered to
  // it, with a jobs failure staying the panel's problem alone.
  useEffect(() => {
    if (!reportOpen || !datasetId) return;
    let cancelled = false;
    void (async () => {
      try {
        const record = await getDatasetV1DatasetsDatasetIdGet(datasetId);
        if (cancelled) return;
        setReportRecord(record);
      } catch (err) {
        if (!cancelled) {
          setReportError(err instanceof ApiError ? err : NETWORK_ERROR);
        }
        return;
      }
      try {
        const listing = await listJobsV1JobsGet();
        if (cancelled) return;
        setReportJobs(
          [...(listing?.jobs ?? [])]
            .filter((job) => job.dataset_id === datasetId)
            .sort((a, b) => b.created_at - a.created_at),
        );
      } catch (err) {
        if (!cancelled) {
          setReportJobsError(err instanceof ApiError ? err : NETWORK_ERROR);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [reportOpen, datasetId]);

  function handleOverridesChange(next: DecisionOverride[]) {
    setQuoteLoading(true);
    setPlanRefusal(null);
    setOverrides(next);
  }

  function handleHyperparametersChange(next: Record<string, string>) {
    setQuoteLoading(true);
    setPlanRefusal(null);
    setHyperparameters(next);
  }

  // In-place edits from the step-2 cards: same commit rule as the surface
  // they replace — an empty or default-equal value clears the override.
  function setHyperValue(name: string, value: string) {
    handleHyperparametersChange({ ...hyperparameters, [name]: value });
  }

  function revertHyperValue(name: string) {
    const next = { ...hyperparameters };
    delete next[name];
    handleHyperparametersChange(next);
  }

  // Back to the smart defaults in one act: clears every pinned decision and
  // every hyperparameter override, and the plan re-requests clean. One state
  // batch, so the quote effect fires once.
  function resetTuning() {
    setQuoteLoading(true);
    setPlanRefusal(null);
    setOverrides([]);
    setHyperparameters({});
  }

  const canContinueSources = preview !== null && !previewLoading;

  // The wizard root is a plain div, deliberately not a form: step 1 embeds
  // the upload/import forms, and a form inside a form is invalid HTML that
  // Next flags as a hydration error. Launching reads component state rather
  // than submitted form data, so nothing is lost.
  async function launch() {
    if (step !== 4) return;
    if (!preview) {
      setRefusal(
        new ApiError(0, "no_dataset", "Choose a dataset to train on first."),
      );
      return;
    }
    // The model radios live on step 1 and unmount on review — the source of
    // truth here is the selected state, not the submitted form data. Reading
    // FormData would find no base_model on step 4 and refuse every launch.
    const chosen = selected;
    if (typeof chosen !== "string" || !chosen) {
      setRefusal(
        new ApiError(0, "no_model", "Choose a base model to train from."),
      );
      return;
    }
    setBusy(true);
    setRefusal(null);
    setStatus("Launching your job…");
    try {
      const job = await createJobV1JobsPost({
        dataset_id: preview.dataset.id,
        base_model: chosen,
        hyperparameters,
        overrides,
        delivery,
      });
      setStatus("Job launched. Opening it…");
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setStatus("");
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  const tunedCount =
    overrides.length + Object.keys(hyperparameters).length;

  const visibleModels = catalog.models;
  const datasetOpts = useMemo(
    () => datasetOptions(preview, localDatasets),
    [preview, localDatasets],
  );
  // Only ready datasets are offered: one that needs fixes cannot start a
  // job, and its report — not this picker — is where it gets fixed. The
  // Datasets page stays one link away for everything else.
  const visibleDatasets = datasetOpts.filter(
    (d) =>
      d.ready &&
      (datasetQuery.trim() === "" ||
        d.filename.toLowerCase().includes(datasetQuery.trim().toLowerCase())),
  );

  const selectedModel =
    catalog.models.find((m) => m.id === selected) ?? null;
  const selectedAdmitted =
    admittedModels.find((a) => a.id === selected) ?? null;
  const methodChosen = quote?.decisions?.find((d) => d.decision === "method");
  const hardwareChosen = quote?.decisions?.find(
    (d) => d.decision === "hardware",
  );
  const precisionChosen =
    quote?.decisions?.find((d) => d.decision === "precision")?.chosen ??
    null;
  const ctxLen =
    selectedModel?.context_length ??
    selectedAdmitted?.probe.context_length ??
    null;
  // The step-2 cards read the same frozen merge the receipt reads: preview
  // hyperparameters with the user's advanced overrides on top, plus any
  // exposed key the preview does not carry, unset.
  const exposedKeys = useMemo(
    () => Object.keys(surface?.tiers.exposed_with_named_failure_mode ?? {}),
    [surface],
  );
  const specGroups = useMemo(
    () =>
      preview ? groupedSpecEntries(preview, hyperparameters, exposedKeys) : [],
    [preview, hyperparameters, exposedKeys],
  );
  // Effective batch is derived, never published: per-device batch times
  // accumulation, so it cannot disagree with either row beside it.
  const effBatch = useMemo(() => {
    if (!preview) return null;
    const h = { ...(preview.hyperparameters ?? {}), ...hyperparameters } as Record<
      string,
      unknown
    >;
    const m = Number(h["micro_batch_size"]);
    const g = Number(h["gradient_accumulation_steps"]);
    return Number.isFinite(m) && Number.isFinite(g) ? m * g : null;
  }, [preview, hyperparameters]);
  // Everything the review step reads: the frozen merge by key, the pinned
  // revision and licence, the validation counts, and the VRAM pair. All
  // null-safe — the review branch renders only with a preview, but these
  // evaluate on every step.
  const specVal = (key: string) =>
    specGroups.find((e) => e.key === key)?.value || "—";
  const reviewReport = preview?.dataset.report ?? null;
  const reviewRev = selectedModel?.revision ?? selectedAdmitted?.revision ?? null;
  const reviewLicence =
    selectedModel?.license ?? selectedAdmitted?.probe.license ?? null;
  const reviewIssues = reviewReport
    ? `${reviewReport.usable_rows.toLocaleString("en-US")} usable rows · ${reviewReport.error_count ?? reviewReport.errors.length} errors, ${reviewReport.warning_count ?? reviewReport.warnings.length} warnings`
    : "Report pending";
  const reviewPeak =
    quote?.peak_memory_gb ?? selectedModel?.peak_memory.total_gb ?? null;
  const reviewCap = selectedModel?.peak_memory.gpu_capacity_gb ?? null;
  const reviewDevices =
    quote?.decisions?.find((d) => d.decision === "device count")?.chosen ??
    "1";
  const reviewGpu =
    quote?.decisions?.find((d) => d.decision === "hardware")?.chosen ??
    selectedModel?.peak_memory.gpu_type ??
    "—";
  const reviewTps = quoteThroughput(quote);
  const hwOverride =
    overrides.find((o) => o.decision === "hardware")?.value ?? null;
  // The default the copy names is the snapshotted predictor default, never
  // the recomputed quote's choice (which echoes a pin back).
  const baseHardware =
    baseChoices["hardware"] ?? hardwareChosen?.chosen ?? "—";
  // The four pre-flight assertions: only client-verifiable state — the
  // selection, the validation report, VRAM arithmetic, estimate presence.
  // A check without data passes open rather than crying wolf: an estimate
  // never refuses a launch, and neither does this banner.
  const preflight = (() => {
    if (!preview) return [];
    const vramPass =
      reviewPeak == null || reviewCap == null || reviewPeak <= reviewCap;
    const vramDetail =
      reviewPeak != null && reviewCap != null
        ? reviewPeak <= reviewCap
          ? `Peak ${reviewPeak.toFixed(1)} / ${reviewCap} GB (+${Math.round(((reviewCap - reviewPeak) / reviewCap) * 100)}% margin)`
          : `Peak ${reviewPeak.toFixed(1)} / ${reviewCap} GB (${Math.round(((reviewPeak - reviewCap) / reviewCap) * 100)}% over)`
        : "Estimate pending";
    return [
      {
        title:
          selectedModel?.repo ?? selectedAdmitted?.repo ?? selected,
        detail: `${reviewRev ? `revision ${reviewRev.slice(0, 12)}` : "no revision"} · ${reviewLicence ?? "unknown licence"}`,
        pass: true,
      },
      {
        title: preview.dataset.filename,
        detail: reviewIssues,
        pass: reviewReport?.valid ?? true,
      },
      { title: "VRAM headroom", detail: vramDetail, pass: vramPass },
      {
        title: "Cost estimate",
        detail: quote
          ? `${formatMinorCostRange(quote.cost_low_minor, quote.cost_high_minor, quote.currency, quote.minor_unit)} · ${formatFriendlyDurationRange(quote.duration_low_s, quote.duration_high_s)}`
          : "No estimate — launching still works",
        pass: quote != null,
      },
    ];
  })();
  const preflightPassed = preflight.filter((p) => p.pass).length;

  return (
    <div className="space-y-6">
      <Stepper step={step} onGo={go} />

      {/* Step-specific heading. Focus moves here on step change so a screen
          reader starts at the top. */}
      <div className="space-y-1">
        <p className="text-[11px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
          Step {step} of 4 — {STEPS[step - 1]?.name}
        </p>
        <h2
          ref={headingRef}
          tabIndex={-1}
          className="text-[22px] leading-tight font-medium tracking-tight outline-none text-balance"
        >
          {step === 1
            ? "Model & dataset"
            : step === 2
              ? "Tune the smart defaults"
              : step === 3
                ? "Choose compute & hardware"
                : "Review and launch"}
        </h2>
        <p className="text-sm text-muted-foreground">
          {step === 1
            ? "Nothing has been spent yet — pick what to train, and what to train from."
            : step === 2
              ? "Temper chose every value below. Change any line and the rest recomputes from the server."
              : step === 3
                ? "Where the job runs, and what it costs."
                : "Everything the job will freeze is on this page. Launching spends money."}
        </p>
      </div>

      {step === 1 && (
        <TooltipProvider delayDuration={0}>
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <section
              aria-labelledby="model-h"
              className="space-y-4 rounded-[12px] border bg-card p-5"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <h3 id="model-h" className="text-xl font-semibold tracking-tight">
                    <span
                      aria-hidden="true"
                      className="mr-2 inline-flex size-6 items-center justify-center rounded-md bg-muted font-mono text-xs text-muted-foreground"
                    >
                      1
                    </span>
                    Select Base Model
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Choose a tested checkpoint, or import weights from outside
                    the catalog.
                  </p>
                </div>
                <div
                  role="group"
                  aria-label="Model source"
                  className="flex shrink-0 rounded-lg bg-muted/50 p-1"
                >
                  {(
                    [
                      { id: "registry", label: "Existing base model" },
                      { id: "import", label: "Import model" },
                    ] as const
                  ).map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      aria-pressed={modelSource === s.id}
                      onClick={() => setModelSource(s.id)}
                      className={cn(
                        "cursor-pointer rounded-md px-3 py-1 text-xs font-medium transition-colors",
                        modelSource === s.id
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {modelSource === "registry" ? (
              <fieldset disabled={busy} id="model-grid" className="scroll-mt-24 space-y-3">
                <legend className="sr-only">Base model</legend>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {catalog.models.map((m) => {
                      const active = m.id === selected;
                      return (
                        <Label
                          key={m.id}
                          htmlFor={`model-${m.id}`}
                          className={cn(
                            "relative flex cursor-pointer items-start gap-3 rounded-lg border bg-card p-4 transition-colors hover:bg-muted/50",
                            active && "border-primary/60 ring-1 ring-primary/40",
                          )}
                        >
                          <Input
                            id={`model-${m.id}`}
                            type="radio"
                            name="base_model"
                            value={m.id}
                            checked={active}
                            onChange={() => onModelChange(m.id)}
                            className="peer sr-only"
                          />
                          {/* The radio is the checkmark: one circular
                              control, checked state shown as a filled check
                              rather than a dot plus a corner badge. The
                              input stays a real radio, so keyboard, focus
                              and screen-reader behavior are unchanged. */}
                          <span
                            aria-hidden="true"
                            className={cn(
                              "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card",
                              active
                                ? "border-primary bg-primary text-primary-foreground"
                                : "border-muted-foreground/40 text-transparent",
                            )}
                          >
                            <Check className="size-3.5" />
                          </span>
                          {/* The description lives behind an info icon with
                              a hover tooltip; screen-reader text stays in
                              the label so the radio's accessible name is
                              unchanged. */}
                          <span className="min-w-0 space-y-1">
                            <span className="block font-medium">
                              {m.repo}
                            </span>
                            <dl className="space-y-1 text-sm text-muted-foreground">
                              <div>
                                <dt className="inline">Licence </dt>
                                <dd className="inline">{m.license}</dd>
                              </div>
                              <div>
                                <dt className="inline">revision </dt>
                                <dd className="inline">
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <code className="cursor-help rounded bg-muted px-1 whitespace-nowrap underline decoration-foreground/30 decoration-dotted underline-offset-4">
                                        <span aria-hidden="true">
                                          {truncateMiddle(m.revision)}
                                        </span>
                                        <span className="sr-only">
                                          {m.revision}
                                        </span>
                                      </code>
                                    </TooltipTrigger>
                                    <TooltipContent className="font-mono">
                                      {m.revision}
                                    </TooltipContent>
                                  </Tooltip>
                                </dd>
                              </div>
                              <div>
                                <dt className="inline">Parameters </dt>
                                <dd className="inline tabular-nums">
                                  {m.params_b}B
                                </dd>
                                <dt className="inline"> · context </dt>
                                <dd className="inline tabular-nums">
                                  {m.context_length}
                                </dd>
                              </div>
                            </dl>
                          </span>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="absolute top-3 right-3 cursor-help">
                                <Info
                                  className="size-4 text-muted-foreground"
                                  aria-hidden="true"
                                />
                                <span className="sr-only">{m.good_for}</span>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>{m.good_for}</TooltipContent>
                          </Tooltip>
                        </Label>
                      );
                    })}
                  </div>
              </fieldset>
              ) : (
              <div className="space-y-4">
                <AdmitModelForm onAdmitted={onAdmitted} />

              {admittedModels.length > 0 && (
                <fieldset className="space-y-3">
                  <legend className="text-lg font-semibold">
                    Imported models
                  </legend>
                  <p className="text-sm text-muted-foreground">
                    These models were admitted from outside the catalog after
                    a compatibility probe. A model that passes with warnings
                    is usable; its findings say exactly what you are taking
                    on.
                  </p>
                  <div className="space-y-3">
                    {admittedModels.map((a) => (
                      <Label
                        key={a.id}
                        htmlFor={`model-${a.id}`}
                        className="flex cursor-pointer items-start gap-3 rounded-lg border bg-card p-4 hover:bg-muted/50 data-[disabled=true]:cursor-not-allowed data-[disabled=true]:opacity-70"
                      >
                        <Input
                          id={`model-${a.id}`}
                          type="radio"
                          name="base_model"
                          value={a.id}
                          disabled={!a.probe.ok}
                          checked={selected === a.id}
                          onChange={() => onModelChange(a.id)}
                          className="peer sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={cn(
                            "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card",
                            selected === a.id
                              ? "border-primary bg-primary text-primary-foreground"
                              : "border-muted-foreground/40 text-transparent",
                          )}
                        >
                          <Check className="size-3.5" />
                        </span>
                        <span className="min-w-0 flex-1 space-y-2">
                          <span className="block font-medium">{a.repo}</span>
                          <dl className="text-sm text-muted-foreground">
                            <div>
                              <dt className="inline">Licence </dt>
                              <dd className="inline">
                                {a.probe.license || "Unknown"}
                              </dd>
                              <dt className="inline"> · revision </dt>
                              <dd className="inline">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <code className="cursor-help rounded bg-muted px-1 whitespace-nowrap underline decoration-foreground/30 decoration-dotted underline-offset-4">
                                      <span aria-hidden="true">
                                        {truncateMiddle(a.revision)}
                                      </span>
                                      <span className="sr-only">
                                        {a.revision}
                                      </span>
                                    </code>
                                  </TooltipTrigger>
                                  <TooltipContent className="font-mono">
                                    {a.revision}
                                  </TooltipContent>
                                </Tooltip>
                              </dd>
                            </div>
                            <div>
                              <dt className="inline">Parameters </dt>
                              <dd className="inline tabular-nums">
                                {(a.probe.params_b ?? 0).toFixed(1)}B
                              </dd>
                              <dt className="inline"> · context </dt>
                              <dd className="inline tabular-nums">
                                {a.probe.context_length}
                              </dd>
                            </div>
                          </dl>
                          <ProbeResultView probe={a.probe} />
                        </span>
                      </Label>
                    ))}
                  </div>
                </fieldset>
              )}
              </div>
              )}
            </section>

            <section
              aria-labelledby="dataset-h"
              className="space-y-4 rounded-[12px] border bg-card p-5"
            >
              <div className="space-y-1">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h3 id="dataset-h" className="text-xl font-semibold tracking-tight">
                    <span
                      aria-hidden="true"
                      className="mr-2 inline-flex size-6 items-center justify-center rounded-md bg-muted font-mono text-xs text-muted-foreground"
                    >
                      2
                    </span>
                    Select Training Dataset
                  </h3>
                  <div
                    role="group"
                    aria-label="Dataset source"
                    className="flex shrink-0 rounded-lg bg-muted/50 p-1"
                  >
                    {(
                      [
                        { id: "existing", label: "Existing dataset" },
                        { id: "add", label: "Upload / Import" },
                      ] as const
                    ).map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        aria-pressed={datasetSource === s.id}
                        onClick={() => setDatasetSource(s.id)}
                      className={cn(
                        "cursor-pointer rounded-md px-3 py-1 text-xs font-medium transition-colors",
                        datasetSource === s.id
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  A verified JSONL corpus. Switching reloads the spec and
                  the estimate further down the wizard.
                </p>
              </div>

              {datasetSource === "add" ? (
                <Tabs defaultValue="upload">
                  <TabsList>
                    <TabsTrigger value="upload">Upload file</TabsTrigger>
                    <TabsTrigger value="import">
                      Import from Hugging Face
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="upload">
                    <UploadForm onUploaded={handleNewDataset} />
                  </TabsContent>
                  <TabsContent value="import">
                    <ImportForm onImported={handleNewDataset} />
                  </TabsContent>
                </Tabs>
              ) : (
                <>
              <div className="flex flex-wrap items-center gap-3">
                <Input
                  type="search"
                  aria-label="Search datasets"
                  placeholder="Search datasets by name…"
                  value={datasetQuery}
                  onChange={(e) => setDatasetQuery(e.target.value)}
                  className="h-9 min-w-52 flex-1"
                />
                {datasetId && (
                  <Dialog
                    open={reportOpen}
                    onOpenChange={(open) => {
                      // Fresh state per opening, set on the interaction
                      // rather than in the fetch effect below.
                      if (open) {
                        setReportRecord(null);
                        setReportError(null);
                        setReportJobs([]);
                        setReportJobsError(null);
                      }
                      setReportOpen(open);
                    }}
                  >
                    <DialogTrigger asChild>
                      <button
                        type="button"
                        className="shrink-0 text-sm font-medium text-primary underline-offset-4 hover:underline"
                      >
                        View validation report
                      </button>
                    </DialogTrigger>
                    <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-[880px]">
                      <DialogTitle className="sr-only">
                        Validation report
                      </DialogTitle>
                      {reportError || !reportRecord ? (
                        <div className="space-y-2 p-2">
                          <p className="font-medium" aria-live="polite">
                            {reportError
                              ? "The report could not be loaded"
                              : "Loading the report…"}
                          </p>
                          {reportError && (
                            <p className="text-sm text-muted-foreground">
                              <code className="rounded bg-muted px-1">
                                {reportError.code}
                              </code>{" "}
                              — {reportError.message}
                            </p>
                          )}
                        </div>
                      ) : (
                        <ReportView
                          record={reportRecord}
                          jobs={reportJobs}
                          jobsError={reportJobsError}
                          showActions={false}
                          stacked
                        />
                      )}
                    </DialogContent>
                  </Dialog>
                )}
              </div>

              <fieldset
                id="dataset-list"
                disabled={busy || previewLoading}
                aria-busy={previewLoading}
                className="scroll-mt-24 space-y-2"
              >
                <legend className="sr-only">Dataset</legend>
                {pendingDataset && (
                  <div
                    aria-live="polite"
                    className="flex items-center gap-3 rounded-lg border border-dashed bg-card/50 p-3"
                  >
                    <Loader2
                      className="size-4 shrink-0 animate-spin text-muted-foreground"
                      aria-hidden="true"
                    />
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-medium">
                        {pendingDataset.filename}
                      </span>
                      <span className="block font-mono text-xs text-muted-foreground">
                        Validating…
                      </span>
                    </span>
                  </div>
                )}
                {visibleDatasets.length === 0 && !pendingDataset ? (
                  <p className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                    {datasetOpts.length === 0
                      ? "No datasets yet — add one with the button above."
                      : datasetQuery.trim() !== ""
                        ? "No dataset matches that search."
                        : "No ready datasets yet — every dataset needs fixes, or add a new one above."}
                  </p>
                ) : (
                  // Two-up like the model grid above: one column of short
                  // rows leaves the whole right half dead at this width.
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {visibleDatasets.map((d) => {
                    const active = d.id === datasetId;
                    return (
                      <Label
                        key={d.id}
                        htmlFor={`dataset-${d.id}`}
                        className={cn(
                          "relative flex cursor-pointer items-start gap-3 rounded-lg border bg-card p-3 transition-colors hover:bg-muted/50",
                          active && "border-primary/60 ring-1 ring-primary/40",
                        )}
                      >
                        <Input
                          id={`dataset-${d.id}`}
                          type="radio"
                          name="dataset"
                          value={d.id}
                          checked={active}
                          onChange={() => onDatasetChange(d.id)}
                          className="peer sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={cn(
                            "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card",
                            active
                              ? "border-primary bg-primary text-primary-foreground"
                              : "border-muted-foreground/40 text-transparent",
                          )}
                        >
                          <Check className="size-3.5" />
                        </span>
                        {/* Same anatomy as the model card: name with a
                            hover tooltip for long names, one fact per line —
                            including the validation summary, which used to
                            hide behind the info icon. Every figure already
                            on the validation report. */}
                        <span className="min-w-0 flex-1 space-y-1">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="block cursor-help truncate text-sm font-medium">
                                <span aria-hidden="true">
                                  {truncateMiddle(d.filename, 18, 16)}
                                </span>
                                <span className="sr-only">{d.filename}</span>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent className="font-mono break-all">
                              {d.filename}
                            </TooltipContent>
                          </Tooltip>
                          {/* Facts run inline and wrap, model-card style: one
                              glance across, not a tall stack with a dead
                              right half. Each label/value pair stays an
                              adjacent dt/dd for screen readers. */}
                          {/* Three rows, each on its own line: usable rows,
                              token count, and the issues status with icon
                              and badge color doing the reading. Each
                              label/value pair stays an adjacent dt/dd for
                              screen readers. */}
                          <dl className="space-y-1 text-sm text-muted-foreground">
                            <div className="flex items-center gap-1.5">
                              <ListChecks
                                className="size-3.5 shrink-0"
                                aria-hidden="true"
                              />
                              <dt className="inline">Usable </dt>
                              <dd className="inline font-medium text-foreground tabular-nums">
                                {d.usableRows != null && d.totalRows != null
                                  ? `${d.usableRows.toLocaleString("en-US")} of ${d.totalRows.toLocaleString("en-US")} rows`
                                  : (d.usableRows != null
                                      ? `${d.usableRows.toLocaleString("en-US")} rows`
                                      : "—")}
                              </dd>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <Hash
                                className="size-3.5 shrink-0"
                                aria-hidden="true"
                              />
                              <dt className="inline">Tokens </dt>
                              <dd className="inline font-medium text-foreground tabular-nums">
                                {d.tokens != null
                                  ? d.tokens.toLocaleString("en-US")
                                  : "—"}
                              </dd>
                            </div>
                            <div className="flex items-center gap-1.5">
                              {d.errors === 0 && d.warnings === 0 ? (
                                <ShieldCheck
                                  className="size-3.5 shrink-0 text-success"
                                  aria-hidden="true"
                                />
                              ) : (
                                <AlertTriangle
                                  className={cn(
                                    "size-3.5 shrink-0",
                                    d.errors > 0
                                      ? "text-destructive"
                                      : "text-warning",
                                  )}
                                  aria-hidden="true"
                                />
                              )}
                              <dt className="inline">Status </dt>
                              <dd className="inline">
                                {d.errors === 0 && d.warnings === 0 ? (
                                  <Badge className="border-success/30 bg-success/10 text-success">
                                    No issues
                                  </Badge>
                                ) : (
                                  <Badge
                                    variant={
                                      d.errors > 0 ? "destructive" : "warning"
                                    }
                                  >
                                    {`${d.errors} error${d.errors === 1 ? "" : "s"} · ${d.warnings} warning${d.warnings === 1 ? "" : "s"}`}
                                  </Badge>
                                )}
                              </dd>
                            </div>
                          </dl>
                        </span>
                      </Label>
                    );
                  })}
                  </div>
                )}
              </fieldset>
              <p aria-live="polite" className="text-sm text-muted-foreground">
                {previewLoading ? "Loading the dataset…" : ""}
              </p>
              {pendingNote && (
                <p role="status" className="text-sm text-muted-foreground">
                  {pendingNote}{" "}
                  <Link
                    href="/datasets"
                    className="font-medium text-primary underline-offset-4 hover:underline"
                  >
                    Open Datasets
                  </Link>
                </p>
              )}
              {previewRefusal && (
                <Alert variant="destructive">
                  <AlertTitle>
                    That dataset cannot start a job.{" "}
                    <code className="rounded bg-muted px-1 text-xs">
                      {previewRefusal.code}
                    </code>
                  </AlertTitle>
                  <AlertDescription>{previewRefusal.message}</AlertDescription>
                </Alert>
              )}
                </>
              )}
            </section>

            {preview && (
              <section
                aria-label="Smart defaults"
                className="space-y-2 rounded-[12px] border bg-card p-5"
              >
                <h3 className="font-medium">
                  Smart Defaults Auto-Calculated
                </h3>
                <SmartDefaultsCopy
                  preview={preview}
                  modelRepo={
                    selectedModel?.repo ?? selectedAdmitted?.repo ?? selected
                  }
                  hardware={hardwareChosen?.chosen}
                  method={methodChosen?.chosen}
                />
              </section>
            )}

            {preview?.warning && (
              // The feasibility estimate reaches the user here rather than
              // after the money starts: this is the last moment they can
              // still act on it — and it reloads with the dataset above.
              <div
                role="alert"
                className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-amber-900"
              >
                <p className="font-semibold">Before you launch</p>
                <p>
                  <code className="rounded bg-amber-100 px-1 text-xs">
                    {preview.warning.code}
                  </code>
                </p>
                <p className="mt-1 text-sm">{preview.warning.message}</p>
              </div>
            )}
          </div>

          <SummaryRail
            step={step}
            preview={preview}
            quote={quote}
            quoteLoading={quoteLoading || previewLoading}
            selectedModel={selectedModel}
            selectedAdmitted={selectedAdmitted}
            hyperparameters={hyperparameters}
            primary={{
              label: "Continue to hyperparameters",
              onClick: () => go(2),
              disabled: !canContinueSources,
            }}
          />
        </div>
        </TooltipProvider>
      )}

      {step === 2 && (
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
          {preview ? (
            <TooltipProvider delayDuration={0}>
              <section
                aria-label="Smart defaults"
                className="space-y-3 rounded-[12px] border bg-card p-5"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    aria-hidden="true"
                    className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
                  >
                    <Sparkles className="size-4 text-primary" />
                  </span>
                  <span className="text-sm font-medium">Smart Defaults</span>
                  <span className="rounded-md border border-primary/30 bg-primary/15 px-2 py-0.5 font-mono text-[11px] font-medium text-primary">
                    Auto-Calculated
                  </span>
                  {precisionChosen && (
                    <span className="ml-auto font-mono text-xs text-muted-foreground">
                      Precision: {precisionChosen}
                    </span>
                  )}
                </div>
                <SmartDefaultsCopy
                  preview={preview}
                  modelRepo={
                    selectedModel?.repo ?? selectedAdmitted?.repo ?? selected
                  }
                  hardware={hardwareChosen?.chosen}
                  method={methodChosen?.chosen}
                />
              </section>

              <section aria-labelledby="spec-h" className="space-y-4">
                <h3 id="spec-h" className="sr-only">
                  The job specification
                </h3>
                <div className="flex flex-wrap items-end justify-between gap-3">
                  <p className="min-w-52 flex-1 text-sm text-muted-foreground">
                    This job will train with the following hyperparameters.
                    They are <strong>frozen at launch</strong>: the job runs
                    with exactly these,{" "}
                    <strong>and they cannot be changed afterwards</strong>.
                    {tunedCount > 0
                      ? ` ${tunedCount} changed by you.`
                      : " All smart defaults."}
                  </p>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={resetTuning}
                    disabled={tunedCount === 0}
                    title={
                      tunedCount === 0
                        ? "Already at the smart defaults"
                        : "Clear every override and return to the smart defaults"
                    }
                  >
                    Reset defaults
                  </Button>
                </div>
                <SpecCard n="01" title="LoRA & Adapter Architecture">
                  {/* The training-shape decisions come first: method and
                      precision govern what rank and alpha below even mean. */}
                  {quote && (
                    <div className="border-b border-border/70 pb-3">
                      <QuoteView
                        quote={quote}
                        editable
                        overrides={overrides}
                        onOverridesChange={handleOverridesChange}
                        refusal={planRefusal}
                        showEstimate={false}
                        showPhases={false}
                        decisions={["method", "precision"]}
                        explain="dialog"
                        bareDecisions
                        defaultChoices={baseChoices}
                      />
                    </div>
                  )}
                  <SpecEntriesList
                    entries={specGroups.filter((e) => e.group === 1)}
                    surface={surface}
                    defaults={preview.hyperparameters ?? {}}
                    overrides={hyperparameters}
                    onSet={setHyperValue}
                    onRevert={revertHyperValue}
                  />
                </SpecCard>

                <SpecCard
                  n="02"
                  title="Training & Optimization"
                  meta={
                    effBatch != null ? (
                      <span className="font-mono text-xs text-muted-foreground tabular-nums">
                        Effective batch = {effBatch}
                      </span>
                    ) : undefined
                  }
                >
                  <SpecEntriesList
                    entries={specGroups.filter((e) => e.group === 2)}
                    surface={surface}
                    defaults={preview.hyperparameters ?? {}}
                    overrides={hyperparameters}
                    onSet={setHyperValue}
                    onRevert={revertHyperValue}
                  />
                </SpecCard>

                <SpecCard
                  n="03"
                  title="Sequence & Context Length"
                  meta={
                    ctxLen != null ? (
                      <span className="font-mono text-xs text-muted-foreground tabular-nums">
                        Model Context Max: {ctxLen.toLocaleString("en-US")}
                      </span>
                    ) : undefined
                  }
                >
                  <SpecEntriesList
                    // The sequence-length decision below is the single control
                    // for this value: the hyperparameter row would duplicate
                    // it (same 2048 twice), so it steps aside while the quote
                    // — and its decision — is present.
                    entries={specGroups.filter(
                      (e) => e.group === 3 && (quote == null || e.key !== "sequence_len"),
                    )}
                    surface={surface}
                    defaults={preview.hyperparameters ?? {}}
                    overrides={hyperparameters}
                    onSet={setHyperValue}
                    onRevert={revertHyperValue}
                  />
                  {quote && (
                    <div className="border-t border-border/70 pt-3">
                      <QuoteView
                        quote={quote}
                        editable
                        overrides={overrides}
                        onOverridesChange={handleOverridesChange}
                        showEstimate={false}
                        showPhases={false}
                        decisions={["sequence length"]}
                        explain="dialog"
                        bareDecisions
                        defaultChoices={baseChoices}
                      />
                    </div>
                  )}
                </SpecCard>

                <SpecCard n="04" title="Evaluation & Checkpointing">
                  <SpecEntriesList
                    entries={specGroups.filter((e) => e.group === 4)}
                    surface={surface}
                    defaults={preview.hyperparameters ?? {}}
                    overrides={hyperparameters}
                    onSet={setHyperValue}
                    onRevert={revertHyperValue}
                  />
                </SpecCard>

                {specGroups.some((e) => e.group === 0) && (
                  <SpecCard n="05" title="Other Settings">
                    <SpecEntriesList
                      entries={specGroups.filter((e) => e.group === 0)}
                      surface={surface}
                      defaults={preview.hyperparameters ?? {}}
                      overrides={hyperparameters}
                      onSet={setHyperValue}
                      onRevert={revertHyperValue}
                    />
                  </SpecCard>
                )}
              </section>

              {!quote && (
                <p
                  aria-live="polite"
                  className="text-sm text-muted-foreground"
                >
                  {quoteLoading
                    ? "Loading the cost and time estimate…"
                    : "A cost and time estimate could not be computed for this model right now. Launching will still work; you just will not see the numbers first."}
                </p>
              )}

              {surface && <AdvancedSurface surface={surface} />}
            </TooltipProvider>
          ) : (
            <NoPreviewPrompt onBack={() => go(1)} />
          )}

          </div>
          <SummaryRail
            step={step}
            preview={preview}
            quote={quote}
            quoteLoading={quoteLoading || previewLoading}
            selectedModel={selectedModel}
            selectedAdmitted={selectedAdmitted}
            hyperparameters={hyperparameters}
            primary={{
              label: "Continue to hardware",
              onClick: () => go(3),
              disabled: !preview,
            }}
            secondary={{ label: "Back to sources", onClick: () => go(1) }}
          />
        </div>
      )}

      {step === 3 && (
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
          {preview && quote ? (
            <>
              <EstimateHero quote={quote} />
              <div className="space-y-3">
                <StepEyebrow n="01">Estimate breakdown</StepEyebrow>
                <QuoteView
                  quote={quote}
                  editable
                  overrides={overrides}
                  onOverridesChange={handleOverridesChange}
                  refusal={planRefusal}
                  showDecisions={false}
                />
              </div>
              <div className="space-y-3">
                <StepEyebrow n="02">Select hardware</StepEyebrow>
                <section
                  aria-label="Select hardware"
                  className="space-y-4 rounded-[12px] border bg-card p-5"
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <p className="text-sm text-muted-foreground">
                      {hwOverride ? (
                        <>
                          Overridden to{" "}
                          <strong className="text-foreground">
                            {hwOverride}
                          </strong>{" "}
                          — Temper&apos;s default was{" "}
                          <strong className="text-foreground">
                            {baseHardware}
                          </strong>
                          .
                        </>
                      ) : (
                        <>
                          Temper&apos;s default for this run is{" "}
                          <strong className="text-foreground">
                            {baseHardware}
                          </strong>{" "}
                          — select another GPU to override it.
                        </>
                      )}
                    </p>
                    {reviewPeak != null && reviewCap != null && (
                      <p className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
                        Peak {reviewPeak.toFixed(1)} / {reviewCap} GB
                      </p>
                    )}
                  </div>
                  <QuoteView
                    quote={quote}
                    editable
                    overrides={overrides}
                    onOverridesChange={handleOverridesChange}
                    refusal={planRefusal}
                    showEstimate={false}
                    showPhases={false}
                    decisions={COMPUTE_DECISIONS}
                    explain="dialog"
                    bareDecisions
                    cardDecisions={["hardware"]}
                    defaultChoices={baseChoices}
                  />
                </section>
              </div>
            </>
          ) : (
            <>
              {preview && (
                <p
                  aria-live="polite"
                  className="text-sm text-muted-foreground"
                >
                  {quoteLoading
                    ? "Loading the cost and time estimate…"
                    : "A cost and time estimate could not be computed for this model right now. Launching will still work; you just will not see the numbers first."}
                </p>
              )}
              {!preview && <NoPreviewPrompt onBack={() => go(1)} />}
            </>
          )}

          </div>
          <SummaryRail
            step={step}
            preview={preview}
            quote={quote}
            quoteLoading={quoteLoading || previewLoading}
            selectedModel={selectedModel}
            selectedAdmitted={selectedAdmitted}
            hyperparameters={hyperparameters}
            primary={{
              label: "Continue to review",
              onClick: () => go(4),
              disabled: !preview,
            }}
            secondary={{
              label: "Back to hyperparameters",
              onClick: () => go(2),
            }}
          />
        </div>
      )}

      {step === 4 && (
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
          {preview ? (
            <>
              {/* Pre-flight: the last thing read before money moves. Every
                  row asserts client-verifiable state with its numbers shown;
                  nothing here refuses a launch. */}
              <section
                aria-label="Pre-flight checks"
                className="space-y-3 rounded-[12px] border bg-card p-5"
              >
                <div className="flex flex-wrap items-center gap-2.5">
                  <span
                    aria-hidden="true"
                    className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
                  >
                    <ShieldCheck className="size-4 text-success" />
                  </span>
                  <span>
                    <span className="block text-[15px] font-semibold tracking-tight">
                      Pre-Flight Checks
                    </span>
                    <span className="block font-mono text-xs text-muted-foreground">
                      {preflight.length} automated assertions against the
                      frozen configuration
                    </span>
                  </span>
                  <span
                    className={cn(
                      "ml-auto rounded-md border px-2 py-0.5 font-mono text-[11px] font-medium",
                      preflightPassed === preflight.length
                        ? "border-success/30 bg-success/10 text-success"
                        : "border-warning/30 bg-warning/10 text-warning",
                    )}
                  >
                    {preflightPassed} of {preflight.length} verified
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {preflight.map((p) => (
                    <PreflightRow
                      key={p.title}
                      pass={p.pass}
                      title={p.title}
                      detail={p.detail}
                    />
                  ))}
                </div>
              </section>

              <ReviewCard
                title="Model & Dataset"
                icon={<Network className="size-4 text-muted-foreground" />}
                editLabel="Edit model and dataset"
                onEdit={() => go(1)}
              >
                <ReviewRow
                  label="Base model"
                  value={
                    selectedModel?.repo ?? selectedAdmitted?.repo ?? selected
                  }
                />
                <ReviewRow
                  label="Revision"
                  value={reviewRev ? reviewRev.slice(0, 12) : "—"}
                  title={reviewRev ?? "—"}
                />
                <ReviewRow
                  label="Dataset"
                  value={preview.dataset.filename}
                />
                <ReviewRow
                  label="Usable rows"
                  value={
                    reviewReport
                      ? reviewReport.usable_rows.toLocaleString("en-US")
                      : "—"
                  }
                />
                <ReviewRow
                  label="Valid tokens"
                  value={
                    reviewReport?.token_count != null
                      ? reviewReport.token_count.toLocaleString("en-US")
                      : "—"
                  }
                />
                <ReviewRow label="Licence" value={reviewLicence ?? "—"} />
              </ReviewCard>

              <ReviewCard
                title="Hyperparameters & Adapter"
                icon={
                  <SlidersHorizontal className="size-4 text-muted-foreground" />
                }
                editLabel="Edit hyperparameters"
                onEdit={() => go(2)}
              >
                <ReviewRow
                  label="Method"
                  value={methodChosen?.chosen ?? "—"}
                />
                <ReviewRow label="Precision" value={precisionChosen ?? "—"} />
                <ReviewRow label="Rank (r)" value={specVal("lora_r")} />
                <ReviewRow label="Alpha" value={specVal("lora_alpha")} />
                <ReviewRow label="Epochs" value={specVal("num_epochs")} />
                <ReviewRow
                  label="Learning rate"
                  value={specVal("learning_rate")}
                />
                <ReviewRow
                  label="Sequence length"
                  value={specVal("sequence_len")}
                />
                <ReviewRow
                  label="Effective batch"
                  value={effBatch != null ? String(effBatch) : "—"}
                />
              </ReviewCard>

              <ReviewCard
                title="Hardware & Estimate"
                icon={<Cpu className="size-4 text-muted-foreground" />}
                editLabel="Edit hardware"
                onEdit={() => go(3)}
              >
                <ReviewRow
                  label="Hardware"
                  value={`${reviewDevices}x ${reviewGpu} GPU`}
                />
                <ReviewRow
                  label="VRAM"
                  value={
                    reviewPeak != null && reviewCap != null
                      ? `${reviewPeak.toFixed(1)} / ${reviewCap} GB`
                      : "—"
                  }
                />
                <ReviewRow
                  label="Est. duration"
                  value={
                    quote
                      ? formatFriendlyDurationRange(
                          quote.duration_low_s,
                          quote.duration_high_s,
                        )
                      : "—"
                  }
                />
                <ReviewRow
                  label="Est. cost"
                  value={
                    quote
                      ? formatMinorCostRange(
                          quote.cost_low_minor,
                          quote.cost_high_minor,
                          quote.currency,
                          quote.minor_unit,
                        )
                      : "—"
                  }
                />
                <ReviewRow
                  label="Changed"
                  value={
                    tunedCount === 0
                      ? "none — smart defaults"
                      : String(tunedCount)
                  }
                />
                <ReviewRow
                  label="Throughput"
                  value={
                    reviewTps != null
                      ? `~${reviewTps.toLocaleString("en-US")} tok/s`
                      : "—"
                  }
                />
              </ReviewCard>

              <details className="group rounded-[12px] border bg-card px-5 py-4">
                <summary className="flex cursor-pointer list-none items-center gap-1.5 font-mono text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
                  <span
                    aria-hidden
                    className="text-[10px] transition-transform group-open:rotate-90"
                  >
                    ▶
                  </span>
                  Full hyperparameters (
                  {specEntries(preview, hyperparameters).length})
                </summary>
                <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3">
                  {specEntries(preview, hyperparameters).map(
                    ([key, value]) => (
                      <div key={key}>
                        <dt className="text-sm text-muted-foreground">
                          <code>{key}</code>
                        </dt>
                        <dd className="font-medium tabular-nums">{value}</dd>
                      </div>
                    ),
                  )}
                </dl>
              </details>

              <fieldset className="space-y-3">
                <legend className="text-lg font-semibold">
                  What you get back
                </legend>
                <p className="text-sm text-muted-foreground">
                  Every job returns the trained artifact. You can also ask
                  for a merged single-file model (to serve it directly) and
                  a quantised local format (to run it on your own machine).
                  These are produced on the machine while it is warm, and
                  each is verified before it ships.
                </p>
                <div className="space-y-2">
                  {deliveryOptions.map((option) => (
                    <Label
                      key={option.id}
                      className="flex items-start gap-3 rounded-lg border bg-card p-3"
                    >
                      <Input
                        type="checkbox"
                        checked={delivery.includes(option.id)}
                        onChange={(e) =>
                          setDelivery(
                            e.target.checked
                              ? [...new Set([...delivery, option.id])]
                              : delivery.filter((d) => d !== option.id),
                          )
                        }
                        className="mt-1 size-4"
                      />
                      <span className="text-sm">
                        <span className="block font-medium">
                          {option.name}
                        </span>
                        <span className="text-muted-foreground">
                          {option.whatFor}
                        </span>
                      </span>
                    </Label>
                  ))}
                </div>
              </fieldset>
            </>
          ) : (
            <NoPreviewPrompt onBack={() => go(1)} />
          )}

          <p aria-live="polite" className="text-sm text-muted-foreground">
            {busy ? status : ""}
          </p>

          {refusal && (
            <Alert variant="destructive">
              <AlertTitle>
                The job could not be launched.{" "}
                <code className="rounded bg-muted px-1 text-xs">
                  {refusal.code}
                </code>
              </AlertTitle>
              <AlertDescription>{refusal.message}</AlertDescription>
            </Alert>
          )}
          </div>
          <SummaryRail
            step={step}
            preview={preview}
            quote={quote}
            quoteLoading={quoteLoading || previewLoading}
            selectedModel={selectedModel}
            selectedAdmitted={selectedAdmitted}
            hyperparameters={hyperparameters}
            primary={{
              label: busy ? "Launching…" : "Launch job",
              onClick: () => void launch(),
              disabled: busy || !preview,
            }}
            secondary={{ label: "Back to hardware", onClick: () => go(3) }}
          />
        </div>
      )}
    </div>
  );
}

// Shown on steps past Sources when there is no dataset yet: reachable only
// by stepping back and clearing, never by advancing, since the Sources
// Continue stays disabled until a preview loads.
function NoPreviewPrompt({ onBack }: { onBack: () => void }) {
  return (
    <div className="space-y-3 rounded-lg border border-dashed bg-card/50 p-6 text-center">
      <p className="font-medium">No dataset picked yet</p>
      <p className="text-sm text-muted-foreground">
        Go back to Sources and pick what this job trains on.
      </p>
      <Button type="button" variant="outline" onClick={onBack}>
        Back to sources
      </Button>
    </div>
  );
}

// The Smart Defaults panel: prose over the preview's own hyperparameters and
// the plan's method/hardware choices — every number on screen already
// appears in the spec or the decisions below, so this explains without
// inventing.
function SmartDefaultsCopy({
  preview,
  modelRepo,
  hardware,
  method,
}: {
  preview: JobSpecPreview;
  modelRepo: string;
  hardware?: string;
  method?: string;
}) {
  const hyper = (preview.hyperparameters ?? {}) as Record<string, unknown>;
  const usable = preview.dataset.report?.usable_rows;
  const str = (v: unknown) => (v === null || v === undefined ? null : String(v));
  const rank = str(hyper["lora_r"]);
  const alpha = str(hyper["lora_alpha"]);
  const epochs = str(hyper["num_epochs"]);
  const lr = str(hyper["learning_rate"]);
  return (
    <p className="text-sm leading-relaxed text-muted-foreground">
      Based on <span className="font-medium text-foreground">{modelRepo}</span>
      {usable != null && (
        <>
          {" "}
          and your{" "}
          <span className="font-medium text-foreground tabular-nums">
            {usable.toLocaleString("en-US")} usable rows
          </span>
        </>
      )}
      , Temper set{" "}
      <span className="font-medium text-foreground">
        {method ?? "a training method"}
      </span>
      {rank && alpha && (
        <>
          {" "}
          with <code className="rounded bg-muted px-1">rank={rank}</code>,{" "}
          <code className="rounded bg-muted px-1">alpha={alpha}</code>
        </>
      )}
      {epochs && <> for {epochs} epochs</>}
      {lr && (
        <>
          {" "}
          at learning rate{" "}
          <span className="font-mono tabular-nums">{lr}</span>
        </>
      )}
      {hardware && (
        <>
          {" "}
          on{" "}
          <span className="font-medium text-foreground">{hardware}</span>
        </>
      )}
      . Change any of it on the next steps.
    </p>
  );
}

// One eyebrow row in the estimate rail: the section name with its live state
// beside it — a green check once its step is complete, otherwise the pill
// that names what is shown (the tuned method, the hardware fit). The active
// step's eyebrow reads at full strength; later sections sit one tier down.
function RailHead({
  eyebrow,
  done,
  active,
  badge,
}: {
  eyebrow: string;
  done: boolean;
  active: boolean;
  badge?: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <h4
        className={cn(
          "text-[11px] font-semibold tracking-[0.14em] uppercase",
          active ? "text-foreground" : "text-muted-foreground",
        )}
      >
        {eyebrow}
        <span className="sr-only">{done ? " (complete)" : ""}</span>
      </h4>
      <span className="flex shrink-0 items-center gap-1.5">
        {badge}
        {done && (
          <span
            aria-hidden="true"
            className="inline-flex size-[18px] items-center justify-center rounded-full bg-success/10 text-success"
          >
            <Check className="size-3" strokeWidth={3} />
          </span>
        )}
      </span>
    </div>
  );
}

// One fact in the rail's two-up grids. The label stays an adjacent dt/dd
// pair, and the value truncates with its exact copy one hover away.
function RailFact({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex min-w-0 items-baseline gap-1.5">
      <dt className="shrink-0 text-[13px] text-muted-foreground">{k}:</dt>
      <dd
        className="truncate font-mono text-[13px] font-medium text-foreground tabular-nums"
        title={v}
      >
        {v}
      </dd>
    </div>
  );
}

// Friendly labels for the hyperparameters the trainer resolves, shown beside
// the raw key the frozen receipt and the journeys read by. Grouped into
// numbered cards — adapter, optimization, sequence, checkpointing — with any
// key the trainer adds later falling into an "Other" card rather than
// vanishing silently.
const SPEC_LABELS: Record<string, string> = {
  lora_r: "LoRA Rank (r)",
  lora_alpha: "LoRA Alpha (α)",
  lora_dropout: "LoRA Dropout",
  learning_rate: "Learning Rate",
  lr_scheduler: "LR Scheduler",
  warmup_ratio: "Warmup Ratio",
  num_epochs: "Epochs",
  max_steps: "Max Steps",
  micro_batch_size: "Per-Device Batch Size",
  gradient_accumulation_steps: "Gradient Accumulation",
  sequence_len: "Max Sequence Length",
  val_set_size: "Validation Split",
  save_total_limit: "Keep Best Checkpoints",
  save_steps: "Save Every",
};

const SPEC_GROUPS: Record<string, number> = {
  lora_r: 1,
  lora_alpha: 1,
  lora_dropout: 1,
  learning_rate: 2,
  lr_scheduler: 2,
  warmup_ratio: 2,
  num_epochs: 2,
  max_steps: 2,
  micro_batch_size: 2,
  gradient_accumulation_steps: 2,
  sequence_len: 3,
  val_set_size: 4,
  save_total_limit: 4,
  save_steps: 4,
};

const SPEC_ORDER = [
  "lora_r",
  "lora_alpha",
  "lora_dropout",
  "learning_rate",
  "lr_scheduler",
  "warmup_ratio",
  "num_epochs",
  "max_steps",
  "micro_batch_size",
  "gradient_accumulation_steps",
  "sequence_len",
  "val_set_size",
  "save_total_limit",
  "save_steps",
];

// Throughput is derived from the quote's own midpoint, so it cannot disagree
// with the duration beside it. One definition for the rail and the step-3
// hero, which show the same figure.
function quoteThroughput(quote: Quote | null): number | null {
  if (
    !quote ||
    quote.token_count == null ||
    quote.duration_low_s <= 0 ||
    quote.duration_high_s <= 0
  )
    return null;
  return Math.round(
    quote.token_count / ((quote.duration_low_s + quote.duration_high_s) / 2),
  );
}

// A numbered eyebrow over a step-3 block, in the mock's language. A plain
// paragraph rather than a heading: the blocks below keep their own headings,
// and doubling them would garble the outline.
function StepEyebrow({ n, children }: { n: string; children: ReactNode }) {
  return (
    <p className="flex items-baseline gap-2.5 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
      <span aria-hidden="true" className="font-mono">
        {n}
      </span>
      <span>{children}</span>
    </p>
  );
}

// The step-3 hero: the friendly duration and the compact cost side by side,
// with the quote's expiry beside the pill. Deliberately the summary tier —
// the precise clock and the per-phase table below stay the figures of record,
// so the two never print the same string twice.
function EstimateHero({ quote }: { quote: Quote }) {
  const tps = quoteThroughput(quote);
  return (
    <section
      aria-label="Estimate summary"
      className="space-y-4 rounded-[12px] border bg-card p-5"
    >
      <div className="flex flex-wrap items-center gap-2">
          <span
            aria-hidden="true"
            className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
          >
            <Cpu className="size-4 text-muted-foreground" />
          </span>
          <span className="rounded-md border border-primary/30 bg-primary/15 px-2 py-0.5 font-mono text-[11px] font-medium text-primary">
            Cost &amp; Time Estimate
          </span>
        <span className="ml-auto font-mono text-xs text-muted-foreground">
          Expires {formatTimestamp(quote.expires_at)}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <p className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
            Est. Duration
          </p>
          <p className="font-mono text-[22px] leading-tight font-semibold tabular-nums">
            {formatFriendlyDurationRange(
              quote.duration_low_s,
              quote.duration_high_s,
            )}
          </p>
          <p className="font-mono text-xs text-muted-foreground tabular-nums">
            {tps != null
              ? `~${tps.toLocaleString("en-US")} tok/s`
              : quote.token_count != null
                ? `${quote.token_count.toLocaleString("en-US")} tokens`
                : "—"}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
            Estimated Cost
          </p>
          <p className="font-mono text-[22px] leading-tight font-semibold tabular-nums">
            {formatMinorCostRange(
              quote.cost_low_minor,
              quote.cost_high_minor,
              quote.currency,
              quote.minor_unit,
            )}
          </p>
          <p className="font-mono text-xs text-muted-foreground tabular-nums">
            {quote.token_count != null
              ? `${quote.token_count.toLocaleString("en-US")} tokens`
              : "—"}
          </p>
        </div>
      </div>
    </section>
  );
}

function groupedSpecEntries(
  preview: JobSpecPreview,
  advanced: Record<string, string>,
  // Exposed keys the preview does not carry (today: max_steps, save_steps)
  // still get a row, unset, rather than losing the launch-accepted control
  // the advanced surface used to host.
  exposedKeys: string[] = [],
): { key: string; value: string; group: number }[] {
  const merged: Record<string, unknown> = {
    ...(preview.hyperparameters ?? {}),
    ...advanced,
  };
  for (const k of exposedKeys) {
    if (!(k in merged)) merged[k] = "";
  }
  const rank = (key: string) => {
    const i = SPEC_ORDER.indexOf(key);
    return i === -1 ? SPEC_ORDER.length : i;
  };
  return Object.entries(merged)
    .map(([key, v]) => ({
      key,
      value: String(v),
      group: SPEC_GROUPS[key] ?? 0,
    }))
    .sort(
      (a, b) =>
        (a.group === 0 ? 99 : a.group) - (b.group === 0 ? 99 : b.group) ||
        rank(a.key) - rank(b.key),
    );
}

// One numbered card on the hyperparameters step. The values are read-only —
// they freeze at launch and only change through the decision controls and
// the advanced surface below — so they read as values, never as inputs.
function SpecCard({
  n,
  title,
  meta,
  children,
}: {
  n: string;
  title: string;
  meta?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={title}
      className="space-y-3 rounded-[12px] border bg-card p-5"
    >
      <div className="flex items-center justify-between gap-2">
        <h4 className="flex items-baseline gap-2.5 text-[15px] font-semibold tracking-tight">
          <span
            aria-hidden="true"
            className="font-mono text-xs font-medium text-muted-foreground"
          >
            {n}
          </span>
          {title}
        </h4>
        {meta}
      </div>
      {children}
    </section>
  );
}

// The frozen rows inside a spec card: friendly label with the raw key beside
// it (the key is what the receipt and the journeys read by), exact value in
// a mono badge. Label/value pairs stay real dt/dd.
function SpecEntriesList({
  entries,
  surface,
  defaults,
  overrides,
  onSet,
  onRevert,
}: {
  entries: { key: string; value: string }[];
  surface: AdvancedSurfaceModel | null;
  defaults: Record<string, unknown>;
  overrides: Record<string, string>;
  onSet: (name: string, value: string) => void;
  onRevert: (name: string) => void;
}) {
  if (entries.length === 0) return null;
  const exposed = surface?.tiers.exposed_with_named_failure_mode ?? {};
  const calculated = surface?.tiers.calculated ?? {};
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
      {entries.map(({ key, value }) => {
        const label = SPEC_LABELS[key] ?? key;
        const field = exposed[key];
        // Only the exposed tier is launch-accepted: a calculated key renders
        // read-only with the platform's reason one hover away, never a
        // control whose value the launch would refuse.
        if (!field) {
          return (
            <ReadOnlySpecRow
              key={key}
              label={label}
              code={key}
              value={value}
              lockedWhy={
                calculated[key]?.reason ?? "Set by the platform."
              }
            />
          );
        }
        const kind =
          field.type === "int" || field.type === "float"
            ? field.type
            : ("string" as const);
        return (
          <EditableSpecRow
            key={key}
            name={key}
            label={label}
            kind={kind}
            defaultValue={defaults[key]}
            override={overrides[key]}
            why={{
              reason: field.reason,
              failureMode: field.failure_mode ?? null,
            }}
            onSet={onSet}
            onRevert={onRevert}
          />
        );
      })}
    </dl>
  );
}

// A read-only frozen row: the value Temper chose, shown, never edited. The
// title names why there is no control — the calculated tier's own reason.
function ReadOnlySpecRow({
  label,
  code,
  value,
  lockedWhy,
}: {
  label: string;
  code: string;
  value: string;
  lockedWhy: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3" title={lockedWhy}>
      <dt className="min-w-0 text-sm font-medium">
        {label !== code ? (
          <>
            {label}{" "}
            <code className="rounded bg-muted px-1 font-mono text-xs font-normal text-muted-foreground">
              {code}
            </code>
          </>
        ) : (
          <code className="font-mono text-sm font-medium">{code}</code>
        )}
      </dt>
      <dd className="shrink-0 rounded-md border bg-muted/60 px-2.5 py-1 font-mono text-[13px] tabular-nums">
        {value === "" ? "—" : value}
      </dd>
    </div>
  );
}

// One hyperparameter edited in place, populated with its effective value.
// Input types are generated from the schema's own `type`, never hand-listed:
// int and float get numeric inputs, everything else text. (The schema
// publishes no bounds, enums or option vocabularies, so sliders, selects and
// toggles would be invented constraints — a slider whose ends the server
// never promised. When the contract grows bounds, this is the seam.)
function EditableSpecRow({
  name,
  label,
  kind,
  defaultValue,
  override,
  why,
  onSet,
  onRevert,
}: {
  name: string;
  label: string;
  kind: "int" | "float" | "string";
  defaultValue: unknown;
  override?: string;
  why: { reason: string; failureMode: string | null };
  onSet: (name: string, value: string) => void;
  onRevert: (name: string) => void;
}) {
  const unset = defaultValue === undefined || defaultValue === null;
  const current = override ?? (unset ? "" : String(defaultValue));
  const changed = override !== undefined;
  // Uncontrolled by design: the input remounts on every effective-value
  // change (a recompute or a revert landed, via `key`), so the DOM is always
  // the draft and there is no state to go stale.
  const inputRef = useRef<HTMLInputElement>(null);

  function commit() {
    const trimmed = (inputRef.current?.value ?? "").trim();
    if (trimmed === "" || trimmed === String(defaultValue ?? "")) {
      onRevert(name);
    } else {
      onSet(name, trimmed);
    }
  }

  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="min-w-0 text-sm font-medium">
        {label}{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs font-normal text-muted-foreground">
          {name}
        </code>{" "}
        {changed && (
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium whitespace-nowrap text-amber-800">
            you changed this
          </span>
        )}
      </dt>
      <dd className="shrink-0">
        <div className="flex items-center gap-1">
          <Input
            ref={inputRef}
            key={override ?? "default"}
            type={kind === "string" ? "text" : "number"}
            step={kind === "int" ? "1" : undefined}
            aria-label={`${name} override`}
            defaultValue={current}
            placeholder={unset && !changed ? "unset" : undefined}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
            }}
            className="h-8 w-28 font-mono text-[13px] tabular-nums"
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={`About ${name}`}
                className="shrink-0 rounded-md p-1 text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
              >
                <CircleHelp className="size-4" aria-hidden="true" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="block w-72 space-y-2 border bg-card px-3.5 py-3 text-left text-[13px] font-normal text-foreground [&>svg]:hidden">
              <p>{why.reason}</p>
              {why.failureMode && (
                <p>
                  <span className="font-medium">What goes wrong: </span>
                  {why.failureMode}
                </p>
              )}
            </TooltipContent>
          </Tooltip>
        </div>
        {changed && (
          <button
            type="button"
            onClick={() => onRevert(name)}
            className="mt-1 text-[11px] text-muted-foreground underline underline-offset-2 hover:text-foreground"
          >
            Use the default
          </button>
        )}
      </dd>
    </div>
  );
}

// One row in a review card: muted label, semibold value truncated with its
// exact copy one hover away. Label/value pairs stay real dt/dd.
function ReviewRow({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="flex min-w-0 items-baseline justify-between gap-3">
      <dt className="shrink-0 text-sm text-muted-foreground">{label}</dt>
      <dd
        className="truncate text-sm font-semibold tabular-nums"
        title={title ?? value}
      >
        {value}
      </dd>
    </div>
  );
}

// One review card on the launch step: icon, title, an Edit button that jumps
// back to the step the card summarises, and a two-up grid of frozen rows.
// The Edit name carries its destination, so three Edit buttons never share
// one accessible name.
function ReviewCard({
  title,
  icon,
  editLabel,
  onEdit,
  children,
}: {
  title: string;
  icon: ReactNode;
  editLabel: string;
  onEdit: () => void;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={title}
      className="space-y-3 rounded-[12px] border bg-card p-5"
    >
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
        >
          {icon}
        </span>
        <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onEdit}
          aria-label={editLabel}
          className="ml-auto text-muted-foreground hover:text-foreground"
        >
          <Pencil className="size-3.5" aria-hidden="true" />
          Edit
        </Button>
      </div>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-2.5 sm:grid-cols-2">
        {children}
      </dl>
    </section>
  );
}

// One pre-flight row: what was asserted, the numbers behind it, and whether
// it held. Only client-verifiable state — selection, validation report,
// VRAM arithmetic, estimate presence — never theater.
function PreflightRow({
  pass,
  title,
  detail,
}: {
  pass: boolean;
  title: string;
  detail: string;
}) {
  return (
    <div className="flex min-w-0 items-start gap-2.5 rounded-lg border bg-muted/40 p-3">
      <span
        aria-hidden="true"
        className={cn(
          "mt-0.5 inline-flex size-[18px] shrink-0 items-center justify-center rounded-full",
          pass ? "bg-success/10 text-success" : "bg-warning/10 text-warning",
        )}
      >
        {pass ? (
          <Check className="size-3" strokeWidth={3} />
        ) : (
          <AlertTriangle className="size-3" />
        )}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-semibold" title={title}>
          {title}
          <span className="sr-only">
            {pass ? " (passed)" : " (needs attention)"}
          </span>
        </span>
        <span
          className="block truncate font-mono text-xs text-muted-foreground tabular-nums"
          title={detail}
        >
          {detail}
        </span>
      </span>
    </div>
  );
}

// The estimate rail beside every step: the same numbers the later steps
// freeze, one glance early, restyled to the Run Estimate & Summary mock —
// eyebrow sections with completion checks, pill badges for the tuned method
// and the hardware fit, two-up facts, and the money at the bottom. Every
// figure reads from the live quote, preview and catalog entry; nothing here
// is a second source, so a value the server did not send reads as a dash
// rather than a guess. Sections check off as their step completes, and the
// step's own actions live here — primary Continue/Launch plus a ghost Back —
// so there is exactly one control of each name per step.
function SummaryRail({
  step,
  preview,
  quote,
  quoteLoading,
  selectedModel,
  selectedAdmitted,
  hyperparameters,
  primary,
  secondary,
}: {
  step: Step;
  preview: JobSpecPreview | null;
  quote: Quote | null;
  quoteLoading: boolean;
  selectedModel: ModelCatalog["models"][number] | null;
  selectedAdmitted: AdmittedModel | null;
  hyperparameters: Record<string, string>;
  primary: { label: string; onClick: () => void; disabled: boolean };
  secondary?: { label: string; onClick: () => void };
}) {
  const hyper = {
    ...(preview?.hyperparameters ?? {}),
    ...hyperparameters,
  } as Record<string, unknown>;
  const str = (v: unknown) =>
    v === null || v === undefined ? null : String(v);
  const asNum = (v: unknown) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const whole = (v: unknown) => {
    const n = asNum(v);
    return n == null ? "—" : Math.round(n).toLocaleString("en-US");
  };
  const compact = (v: unknown) => {
    const n = asNum(v);
    return n == null
      ? "—"
      : new Intl.NumberFormat("en-US", {
          notation: "compact",
          maximumFractionDigits: 1,
        }).format(n);
  };
  // A learning rate reads like the mock: 0.0002 is "2e-4".
  const lr = (() => {
    const raw = hyper["learning_rate"];
    const n = asNum(raw);
    if (n == null) return null;
    if (n !== 0 && Math.abs(n) < 0.01)
      return n.toExponential(0).replace("e-0", "e-").replace("e+0", "e+");
    return String(raw);
  })();

  const modelRepo =
    selectedModel?.repo ?? selectedAdmitted?.repo ?? "No model picked yet";
  const paramsB =
    selectedModel?.params_b ?? selectedAdmitted?.probe.params_b ?? null;
  const ctxLen =
    selectedModel?.context_length ??
    selectedAdmitted?.probe.context_length ??
    null;
  const licence =
    selectedModel?.license ?? selectedAdmitted?.probe.license ?? null;

  const decision = (name: string) =>
    quote?.decisions?.find((d) => d.decision === name)?.chosen ?? null;
  const method = decision("method");
  const precision = decision("precision");
  const hardware = decision("hardware");
  const devices = decision("device count") ?? "1";

  const report = preview?.dataset.report ?? null;
  const gpuType = hardware ?? selectedModel?.peak_memory.gpu_type ?? "—";
  const gpuCapacity = selectedModel?.peak_memory.gpu_capacity_gb ?? null;
  const peakGb =
    quote?.peak_memory_gb ?? selectedModel?.peak_memory.total_gb ?? null;
  const vramPct =
    peakGb != null && gpuCapacity
      ? Math.min(100, (peakGb / gpuCapacity) * 100)
      : null;
  // The envelope breakdown behind the bar: the catalog's own per-component
  // prediction, scaled against the same capacity. Admitted models carry no
  // breakdown (their probe records a peak only), so they keep the single bar.
  const mem = selectedModel?.peak_memory ?? null;
  const vramSegs =
    mem && gpuCapacity
      ? [
          { label: "Weights", gb: mem.weights_gb, cls: "bg-primary" },
          { label: "Act", gb: mem.activations_gb, cls: "bg-primary/60" },
          {
            label: "Opt",
            gb: mem.optimizer_gb + mem.gradients_gb,
            cls: "bg-primary/35",
          },
          { label: "Overhead", gb: mem.overhead_gb, cls: "bg-primary/20" },
        ].map((s) => ({ ...s, pct: (s.gb / gpuCapacity) * 100 }))
      : null;
  // Trainable share of the base model, from the catalog's own counts.
  const trainable = selectedModel?.peak_memory.trainable_params ?? null;
  const trainablePct =
    trainable != null && paramsB
      ? (trainable / (paramsB * 1e9)) * 100
      : null;
  // The fit pill is a reading of two published numbers — the predictor's peak
  // against the chosen card's capacity — not a second opinion. "Optimal"
  // while four-fifths headroom remains, "Tight" once past it.
  const fit =
    peakGb != null && gpuCapacity
      ? peakGb <= gpuCapacity * 0.8
        ? "Optimal"
        : peakGb <= gpuCapacity
          ? "Tight"
          : "Exceeds"
      : null;
  // Throughput is derived from the quote's own midpoint, so it cannot
  // disagree with the duration beside it.
  const tps = quoteThroughput(quote);

  const foundationDone =
    selectedModel !== null || selectedAdmitted !== null;
  const corpusDone = preview !== null;
  const tuningDone = step >= 3;
  const hardwareDone = step >= 4;

  return (
    <aside
      aria-label="Run estimate and summary"
      className="space-y-4 rounded-[12px] border bg-card p-5 lg:sticky lg:top-4"
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[15px] font-semibold tracking-tight">
          Run Estimate &amp; Summary
        </h3>
        <span className="rounded-md border bg-muted/60 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
          Step {step} of 4
        </span>
      </div>

      <section
        aria-label="Foundation architecture"
        className="space-y-2.5 border-t border-border/70 pt-4"
      >
        <RailHead
          eyebrow="Foundation architecture"
          done={foundationDone}
          active={step === 1}
        />
        <div className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
          >
            <Network className="size-4 text-muted-foreground" />
          </span>
          <p
            className="truncate text-sm font-semibold"
            title={modelRepo}
          >
            {modelRepo}
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <RailFact k="Parameters" v={paramsB != null ? `${paramsB}B` : "—"} />
          <RailFact
            k="Context"
            v={ctxLen != null ? ctxLen.toLocaleString("en-US") : "—"}
          />
          <RailFact
            k="Precision"
            v={precision ?? (quoteLoading ? "…" : "—")}
          />
          <RailFact k="Licence" v={licence ?? "—"} />
        </dl>
      </section>

      <section
        aria-label="Training corpus"
        className="space-y-2.5 border-t border-border/70 pt-4"
      >
        <RailHead
          eyebrow="Training corpus"
          done={corpusDone}
          active={step === 1}
        />
        <div className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
          >
            <Database className="size-4 text-muted-foreground" />
          </span>
          <p
            className="truncate text-sm font-semibold"
            title={preview?.dataset.filename ?? "No dataset picked yet"}
          >
            {preview ? preview.dataset.filename : "No dataset picked yet"}
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <RailFact k="Samples" v={report ? whole(report.usable_rows) : "—"} />
          <RailFact k="Rows" v={report ? whole(report.row_count) : "—"} />
          <RailFact k="Schema" v={report?.schema_type ?? "—"} />
          <RailFact
            k="Valid tokens"
            v={report?.token_count != null ? compact(report.token_count) : "—"}
          />
        </dl>
      </section>

      {/* Tuning is step 2's subject: it joins the rail once its step is
          reached rather than previewing dimmed on step 1. */}
      {step >= 2 && (
      <section
        aria-label="Tuning mode"
        className="space-y-2.5 border-t border-border/70 pt-4"
      >
        <RailHead
          eyebrow="Tuning mode"
          done={tuningDone}
          active={step === 2}
          badge={
            method ? (
              <span className="rounded-md border border-primary/30 bg-primary/15 px-2 py-0.5 font-mono text-[11px] font-medium text-primary">
                {method}
              </span>
            ) : undefined
          }
        />
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <RailFact k="Rank (r)" v={str(hyper["lora_r"]) ?? "—"} />
          <RailFact k="Alpha" v={str(hyper["lora_alpha"]) ?? "—"} />
          <RailFact k="Epochs" v={str(hyper["num_epochs"]) ?? "—"} />
          <RailFact k="Learning Rate" v={lr ?? "—"} />
          <RailFact
            k="Batch Size"
            v={str(hyper["micro_batch_size"]) ?? "—"}
          />
          <RailFact
            k="Grad Accum"
            v={str(hyper["gradient_accumulation_steps"]) ?? "—"}
          />
        </dl>
        {trainable != null && paramsB != null && (
          <p
            className="truncate font-mono text-xs text-muted-foreground tabular-nums"
            title={`${trainable.toLocaleString("en-US")} trainable parameters`}
          >
            {compact(trainable)} trainable / {paramsB}B total
            {trainablePct != null ? ` · ${trainablePct.toFixed(2)}%` : ""}
          </p>
        )}
      </section>
      )}

      <section
        aria-label="Hardware target"
        className="space-y-2.5 border-t border-border/70 pt-4"
      >
        <RailHead
          eyebrow="Hardware target"
          done={hardwareDone}
          active={step === 3}
          badge={
            fit ? (
              <span
                className={cn(
                  "rounded-md border px-2 py-0.5 font-mono text-[11px] font-medium",
                  fit === "Optimal" &&
                    "border-success/30 bg-success/10 text-success",
                  fit === "Tight" &&
                    "border-warning/30 bg-warning/10 text-warning",
                  fit === "Exceeds" &&
                    "border-destructive/30 bg-destructive/10 text-destructive",
                )}
              >
                {fit}
              </span>
            ) : undefined
          }
        />
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              aria-hidden="true"
              className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-muted/60"
            >
              <Cpu className="size-4 text-muted-foreground" />
            </span>
            <p
              className="truncate text-sm font-semibold"
              title={`${devices}x ${gpuType} GPU`}
            >
              {devices}x {gpuType} GPU
            </p>
          </div>
          <p className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
            {gpuCapacity != null
              ? `${Number.isInteger(gpuCapacity) ? gpuCapacity : gpuCapacity.toFixed(1)} GB VRAM`
              : ""}
          </p>
        </div>
        <dl className="space-y-1.5">
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-[13px] text-muted-foreground">
              Est. Duration:
            </dt>
            <dd className="font-mono text-[13px] font-medium tabular-nums">
              {quote
                ? formatFriendlyDurationRange(
                    quote.duration_low_s,
                    quote.duration_high_s,
                  )
                : quoteLoading
                  ? "…"
                  : "—"}
            </dd>
          </div>
        </dl>
        {vramSegs ? (
          <>
            <div
              role="progressbar"
              aria-label="Predicted VRAM breakdown against card capacity"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={
                vramPct != null ? Math.round(vramPct) : undefined
              }
              className="flex h-1.5 overflow-hidden rounded-full bg-muted-foreground/25"
            >
              {vramSegs.map((s) => (
                <div
                  key={s.label}
                  aria-hidden="true"
                  title={`${s.label} ${s.gb.toFixed(1)} GB`}
                  className={cn("h-full", s.cls)}
                  style={{ width: `${s.pct}%` }}
                />
              ))}
            </div>
            <p className="truncate font-mono text-[11px] text-muted-foreground tabular-nums">
              {vramSegs.map((s) => `${s.label} ${s.gb.toFixed(1)}`).join(" · ")}{" "}
              GB
            </p>
          </>
        ) : (
          vramPct != null && (
            <div
              role="progressbar"
              aria-label="Predicted peak VRAM against card capacity"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(vramPct)}
              className="h-1.5 overflow-hidden rounded-full bg-muted-foreground/25"
            >
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${vramPct}%` }}
              />
            </div>
          )
        )}
        <dl className="flex items-baseline justify-between gap-2">
          <div className="flex items-baseline gap-1.5 font-mono text-xs tabular-nums">
            <dt className="font-sans text-xs text-muted-foreground">
              Tokens/sec:
            </dt>
            <dd>
              {tps != null ? `~${tps.toLocaleString("en-US")}` : "—"}
            </dd>
          </div>
          <div className="flex items-baseline gap-1.5 font-mono text-xs tabular-nums">
            <dt className="font-sans text-xs text-muted-foreground">
              Max VRAM peak:
            </dt>
            <dd>
              {peakGb != null
                ? gpuCapacity != null
                  ? `${peakGb.toFixed(1)} / ${Number.isInteger(gpuCapacity) ? gpuCapacity : gpuCapacity.toFixed(1)} GB`
                  : `${peakGb.toFixed(1)} GB`
                : "—"}
            </dd>
          </div>
        </dl>
      </section>

      <div className="border-t border-border/70 pt-4">
        <p
          className={cn(
            "text-[11px] font-semibold tracking-[0.14em] uppercase",
            step === 4 ? "text-foreground" : "text-muted-foreground",
          )}
        >
          Estimated Cost
        </p>
        <p className="mt-1 font-mono text-[22px] leading-tight font-semibold tabular-nums">
          {quote
            ? formatMinorCostRange(
                quote.cost_low_minor,
                quote.cost_high_minor,
                quote.currency,
                quote.minor_unit,
              )
            : quoteLoading
              ? "…"
              : "—"}
        </p>
      </div>

      <Button
        type="button"
        className="w-full"
        size="lg"
        onClick={primary.onClick}
        disabled={primary.disabled}
      >
        {primary.label}
        <ArrowRight className="size-4" aria-hidden="true" />
      </Button>
      {secondary && (
        <Button
          type="button"
          variant="ghost"
          className="w-full"
          onClick={secondary.onClick}
        >
          <ArrowLeft className="size-4" aria-hidden="true" />
          {secondary.label}
        </Button>
      )}
    </aside>
  );
}

// Long identifiers show their two ends with the middle cut: a pinned
// revision reads the same at a glance either way, and a long import path
// keeps its origin and its file. Ellipsized text must never be the only
// copy — a hover tooltip and screen-reader text keep the exact value one
// hover away.
function truncateMiddle(text: string, head = 6, tail = 6): string {
  return text.length > head + tail + 1
    ? `${text.slice(0, head)}…${text.slice(-tail)}`
    : text;
}

function specEntries(  preview: JobSpecPreview,
  advanced: Record<string, string>,
): [string, string][] {
  const merged = { ...(preview.hyperparameters ?? {}), ...advanced };
  return Object.entries(merged).map(([k, v]) => [k, String(v)]);
}

// The dataset radios on the Sources step: every dataset the server listed,
// ready ones selectable inline. The current preview's own dataset is merged
// in (it covers a list fetched before it existed), and the order is always
// newest-first — never re-sorted around the selection, so picking one never
// shifts the grid under the cursor.
function datasetOptions(
  preview: JobSpecPreview | null,
  datasets: DatasetRecord[],
): {
  id: string;
  filename: string;
  usableRows: number | null;
  totalRows: number | null;
  tokens: number | null;
  errors: number;
  warnings: number;
  ready: boolean;
}[] {
  const seen = new Set<string>();
  const rows: {
    id: string;
    filename: string;
    usableRows: number | null;
    totalRows: number | null;
    tokens: number | null;
    errors: number;
    warnings: number;
    ready: boolean;
    createdAt: number;
  }[] = [];
  if (preview) {
    const report = preview.dataset.report;
    const current = {
      id: preview.dataset.id,
      filename: preview.dataset.filename,
      usableRows: report?.usable_rows ?? null,
      totalRows: report?.row_count ?? null,
      tokens: report?.token_count ?? null,
      errors: report?.error_count ?? report?.errors.length ?? 0,
      warnings: report?.warning_count ?? report?.warnings.length ?? 0,
      ready: true,
      createdAt: preview.dataset.created_at,
    };
    rows.push(current);
    seen.add(preview.dataset.id);
  }
  for (const d of datasets) {
    if (seen.has(d.id)) continue;
    seen.add(d.id);
    rows.push({
      id: d.id,
      filename: d.filename,
      usableRows: d.report?.usable_rows ?? null,
      totalRows: d.report?.row_count ?? null,
      tokens: d.report?.token_count ?? null,
      errors: d.report?.error_count ?? d.report?.errors.length ?? 0,
      warnings: d.report?.warning_count ?? d.report?.warnings.length ?? 0,
      ready: d.report?.valid ?? false,
      createdAt: d.created_at,
    });
  }
  rows.sort((a, b) => b.createdAt - a.createdAt);
  return rows;
}
