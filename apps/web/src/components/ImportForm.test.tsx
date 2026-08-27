import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ImportForm from "@/components/ImportForm";

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

// The generated client is the seam: these tests prove what the user does and
// sees, with the transport mocked out. The real client is exercised by the
// Playwright journeys.
const importMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", () => ({
  importDatasetV1DatasetsImportPost: importMock,
}));

import { ApiError } from "@/lib/api/mutator";

function fillRepo(user: ReturnType<typeof userEvent.setup>, repo = "acme/demo-chat") {
  return user.type(screen.getByLabelText("Public repository"), repo);
}

beforeEach(() => {
  importMock.mockReset();
  push.mockReset();
});

describe("ImportForm", () => {
  it("offers a labelled repository control and a named submit action", () => {
    render(<ImportForm />);
    expect(screen.getByLabelText("Public repository")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Import and validate" }),
    ).toBeVisible();
  });

  it("sends the reference through the generated client and opens the report", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await fillRepo(user, "open-r1/OpenR1-Math-220k");
    await user.type(screen.getByLabelText("Configuration (optional)"), "sft");
    await user.type(screen.getByLabelText("Split (optional)"), "train");
    importMock.mockResolvedValueOnce({ id: "ds_abc", filename: "open-r1/OpenR1-Math-220k/sft/train.jsonl", status: "validating" });
    await user.click(screen.getByRole("button", { name: "Import and validate" }));

    expect(importMock).toHaveBeenCalledWith({
      repo: "open-r1/OpenR1-Math-220k",
      config: "sft",
      split: "train",
    });
    expect(push).toHaveBeenCalledWith("/datasets/ds_abc");
  });

  it("sends just the repository when configuration and split are empty", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await fillRepo(user);
    importMock.mockResolvedValueOnce({ id: "ds_abc", filename: "acme/demo-chat/default/train.jsonl", status: "validating" });
    await user.click(screen.getByRole("button", { name: "Import and validate" }));

    expect(importMock).toHaveBeenCalledWith({ repo: "acme/demo-chat" });
  });

  it("refuses to submit without a repository, naming the problem", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await user.click(screen.getByRole("button", { name: "Import and validate" }));
    expect(importMock).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/repository/i);
  });

  it("shows a refused import's stable code and message", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await fillRepo(user, "nope/nowhere");
    importMock.mockRejectedValueOnce(
      new ApiError(
        400,
        "repo_not_found",
        "the repository 'nope/nowhere' does not exist or is not public",
      ),
    );
    await user.click(screen.getByRole("button", { name: "Import and validate" }));

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("repo_not_found");
    expect(alert).toHaveTextContent("nope/nowhere");
    expect(push).not.toHaveBeenCalled();
  });

  it("survives an unreachable server without losing the refusal", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await fillRepo(user);
    importMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await user.click(screen.getByRole("button", { name: "Import and validate" }));
    expect(screen.getByRole("alert")).toHaveTextContent("network_error");
  });

  it("disables the action while the import is in flight", async () => {
    const user = userEvent.setup();
    render(<ImportForm />);
    await fillRepo(user);
    let resolve!: (v: unknown) => void;
    importMock.mockReturnValueOnce(
      new Promise((res) => {
        resolve = res;
      }),
    );
    await user.click(screen.getByRole("button", { name: "Import and validate" }));
    expect(screen.getByRole("button", { name: /importing/i })).toBeDisabled();
    resolve({});
  });
});
