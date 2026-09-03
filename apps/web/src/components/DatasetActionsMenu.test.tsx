import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DatasetActionsMenu from "@/components/DatasetActionsMenu";

const refresh = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh }),
}));

const deleteMock = vi.hoisted(() => vi.fn());
const renameMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", () => ({
  deleteDatasetV1DatasetsDatasetIdDelete: deleteMock,
  renameDatasetV1DatasetsDatasetIdPatch: renameMock,
}));

import { ApiError } from "@/lib/api/mutator";

beforeEach(() => {
  deleteMock.mockReset();
  renameMock.mockReset();
  refresh.mockReset();
});

async function openMenu(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    screen.getByRole("button", { name: "Actions for acme/demo.jsonl" }),
  );
}

describe("DatasetActionsMenu: rename", () => {
  it("opens with the current name pre-filled and the extension fixed", async () => {
    const user = userEvent.setup();
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /rename/i }));

    // The extension is not part of the editable input -- it rides along as
    // fixed text beside it, so a rename can never produce an unsupported
    // extension (feedback: renaming changed ".jsonl" to whatever was typed).
    expect(screen.getByLabelText("Filename")).toHaveValue("acme/demo");
    expect(screen.getByText(".jsonl")).toBeVisible();
  });

  it("saves the new name with the original extension and refreshes the list", async () => {
    const user = userEvent.setup();
    renameMock.mockResolvedValueOnce({});
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /rename/i }));
    const input = screen.getByLabelText("Filename");
    await user.clear(input);
    await user.type(input, "renamed");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(renameMock).toHaveBeenCalledWith("ds_abc", {
      filename: "renamed.jsonl",
    });
    expect(refresh).toHaveBeenCalled();
  });

  it("disables Save when the name is unchanged or empty", async () => {
    const user = userEvent.setup();
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /rename/i }));

    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();

    const input = screen.getByLabelText("Filename");
    await user.clear(input);
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("shows a refusal's stable code without closing", async () => {
    const user = userEvent.setup();
    renameMock.mockRejectedValueOnce(
      new ApiError(404, "not_found", "No such dataset."),
    );
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /rename/i }));
    const input = screen.getByLabelText("Filename");
    await user.clear(input);
    await user.type(input, "renamed");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("not_found");
    expect(refresh).not.toHaveBeenCalled();
    expect(
      screen.getByRole("heading", { name: "Rename dataset" }),
    ).toBeVisible();
  });
});

describe("DatasetActionsMenu: delete", () => {
  it("asks for confirmation before deleting", async () => {
    const user = userEvent.setup();
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /delete/i }));

    expect(
      screen.getByRole("heading", { name: "Delete this dataset?" }),
    ).toBeVisible();
    expect(deleteMock).not.toHaveBeenCalled();
  });

  it("deletes and refreshes the list on confirm", async () => {
    const user = userEvent.setup();
    deleteMock.mockResolvedValueOnce(undefined);
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: "Delete" }));

    expect(deleteMock).toHaveBeenCalledWith("ds_abc");
    expect(refresh).toHaveBeenCalled();
  });

  it("cancels without deleting", async () => {
    const user = userEvent.setup();
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(deleteMock).not.toHaveBeenCalled();
    expect(
      screen.queryByRole("heading", { name: "Delete this dataset?" }),
    ).not.toBeInTheDocument();
  });

  it("shows a refusal's stable code and message without closing", async () => {
    const user = userEvent.setup();
    deleteMock.mockRejectedValueOnce(
      new ApiError(
        409,
        "dataset_in_use",
        "This dataset was used by 1 job and cannot be deleted.",
      ),
    );
    render(<DatasetActionsMenu id="ds_abc" filename="acme/demo.jsonl" />);

    await openMenu(user);
    await user.click(screen.getByRole("menuitem", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: "Delete" }));

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("dataset_in_use");
    expect(alert).toHaveTextContent("cannot be deleted");
    expect(refresh).not.toHaveBeenCalled();
    expect(
      screen.getByRole("heading", { name: "Delete this dataset?" }),
    ).toBeVisible();
  });
});
