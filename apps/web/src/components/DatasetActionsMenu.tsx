"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { MoreVertical, Pencil, Trash2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import {
  deleteDatasetV1DatasetsDatasetIdDelete,
  renameDatasetV1DatasetsDatasetIdPatch,
} from "@/lib/api/generated/client";
import { ApiError, NETWORK_ERROR } from "@/lib/api/mutator";

// The product only ever ingests JSONL (`datasets._check_extension`, the same
// guard the backend enforces on this and on upload) -- the extension is not
// something a rename should let a user type wrong, so the input edits only
// the stem and the extension rides along fixed. Order matters: ".jsonl"
// does not end with ".json", so checking the longer one is unnecessary here,
// but listing it first keeps the search obviously correct rather than
// incidentally correct.
const JSONL_EXTENSIONS = [".jsonl", ".json"];

function splitExtension(filename: string): {
  stem: string;
  extension: string;
} {
  const extension = JSONL_EXTENSIONS.find((e) => filename.endsWith(e)) ?? "";
  return {
    stem: filename.slice(0, filename.length - extension.length),
    extension,
  };
}

// The two actions a dataset can be acted on beyond create/read (rename,
// delete) live behind one kebab menu rather than inline buttons next to the
// name -- a second or third click target there competed with the filename
// for attention. Each opens its own confirmation dialog: rename because a
// typo shouldn't need a second trip, delete because there is no undo (the
// API's own contract) and a browser confirm() is something a fast
// double-click can race past. Both dialogs are separate Radix roots from the
// menu, opened via onSelect with the item's default (auto-close-only)
// behaviour prevented -- the documented way to chain a menu into a dialog
// without the menu's focus-trap teardown fighting the dialog's.
export default function DatasetActionsMenu({
  id,
  filename,
  redirectOnDeleteTo,
}: {
  id: string;
  filename: string;
  /** The list view deletes in place (the row just leaves the grid on
      refresh); the detail view has nothing left to render once its own
      dataset is gone, so it navigates away instead. Omit for the former. */
  redirectOnDeleteTo?: string;
}) {
  const router = useRouter();
  const { stem: originalStem, extension } = splitExtension(filename);
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const [stem, setStem] = useState(originalStem);

  async function onRename() {
    setBusy(true);
    setRefusal(null);
    try {
      await renameDatasetV1DatasetsDatasetIdPatch(id, {
        filename: stem.trim() + extension,
      });
      setRenameOpen(false);
      router.refresh();
    } catch (err) {
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  async function onDelete() {
    setBusy(true);
    setRefusal(null);
    try {
      await deleteDatasetV1DatasetsDatasetIdDelete(id);
      setDeleteOpen(false);
      if (redirectOnDeleteTo) {
        router.push(redirectOnDeleteTo);
      } else {
        router.refresh();
      }
    } catch (err) {
      setRefusal(err instanceof ApiError ? err : NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={`Actions for ${filename}`}
            className="text-muted-foreground"
          >
            <MoreVertical aria-hidden="true" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              setStem(originalStem);
              setRefusal(null);
              setRenameOpen(true);
            }}
          >
            <Pencil aria-hidden="true" />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem
            variant="destructive"
            onSelect={(e) => {
              e.preventDefault();
              setRefusal(null);
              setDeleteOpen(true);
            }}
          >
            <Trash2 aria-hidden="true" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename dataset</DialogTitle>
          </DialogHeader>

          <div className="space-y-2">
            <Label htmlFor="dataset-rename-input">Filename</Label>
            <div className="flex h-9 items-stretch overflow-hidden rounded-md border border-input bg-muted/50 has-[input:focus-visible]:border-ring has-[input:focus-visible]:ring-3 has-[input:focus-visible]:ring-ring/50">
              <input
                id="dataset-rename-input"
                type="text"
                value={stem}
                onChange={(e) => setStem(e.target.value)}
                className="min-w-0 flex-1 bg-transparent px-3 text-base text-foreground placeholder:text-muted-foreground outline-none md:text-sm"
                autoFocus
              />
              <span className="flex shrink-0 items-center border-l border-input px-3 text-sm text-muted-foreground select-none">
                {extension}
              </span>
            </div>
          </div>

          {refusal && (
            <Alert variant="destructive">
              <AlertTitle>
                The rename was refused.{" "}
                <code className="rounded bg-muted px-1 text-xs">
                  {refusal.code}
                </code>
              </AlertTitle>
              <AlertDescription>{refusal.message}</AlertDescription>
            </Alert>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setRenameOpen(false)}
              disabled={busy}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={onRename}
              disabled={busy || stem.trim() === "" || stem === originalStem}
            >
              {busy ? "Saving…" : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this dataset?</DialogTitle>
            <DialogDescription>
              <code className="rounded bg-muted px-1 text-xs">{filename}</code>{" "}
              will be removed, along with its stored file. This cannot be
              undone.
            </DialogDescription>
          </DialogHeader>

          {refusal && (
            <Alert variant="destructive">
              <AlertTitle>
                The delete was refused.{" "}
                <code className="rounded bg-muted px-1 text-xs">
                  {refusal.code}
                </code>
              </AlertTitle>
              <AlertDescription>{refusal.message}</AlertDescription>
            </Alert>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setDeleteOpen(false)}
              disabled={busy}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={onDelete}
              disabled={busy}
            >
              {busy ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
