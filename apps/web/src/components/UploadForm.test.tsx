import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import UploadForm from "@/components/UploadForm";

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

// The generated client is the seam: these tests prove what the user does and
// sees, with the transport mocked out. The real client is exercised by the
// Playwright journeys.
const uploadMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", () => ({
  uploadDatasetV1DatasetsPost: uploadMock,
}));

import { ApiError } from "@/lib/api/mutator";

function chooseFile(user: ReturnType<typeof userEvent.setup>, name = "d.jsonl") {
  const input = screen.getByLabelText("Dataset file (.jsonl)");
  const file = new File(['{"messages": []}'], name, { type: "application/json" });
  return user.upload(input, file);
}

beforeEach(() => {
  uploadMock.mockReset();
  push.mockReset();
});

describe("UploadForm", () => {
  it("offers a labelled file control and a named submit action", () => {
    render(<UploadForm />);
    expect(screen.getByLabelText("Dataset file (.jsonl)")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Upload and validate" }),
    ).toBeVisible();
  });

  it("sends the file through the generated client and opens the report", async () => {
    const user = userEvent.setup();
    render(<UploadForm />);
    await chooseFile(user);
    uploadMock.mockResolvedValueOnce({ id: "ds_abc" });
    await user.click(screen.getByRole("button", { name: "Upload and validate" }));
    expect(uploadMock).toHaveBeenCalledTimes(1);
    expect(push).toHaveBeenCalledWith("/datasets/ds_abc");
  });

  it("keeps the action disabled until a file is chosen", async () => {
    const user = userEvent.setup();
    render(<UploadForm />);
    const button = screen.getByRole("button", { name: "Upload and validate" });
    expect(button).toBeDisabled();
    expect(uploadMock).not.toHaveBeenCalled();
    await chooseFile(user);
    expect(button).toBeEnabled();
  });

  it("shows a refused upload's stable code and message", async () => {
    // A file that passes the picker (accept=".jsonl,.json") can still be
    // refused by the API -- oversized, for instance. The refusal renders
    // here with its stable code, exactly as the API stated it.
    const user = userEvent.setup();
    render(<UploadForm />);
    await chooseFile(user);
    uploadMock.mockRejectedValueOnce(
      new ApiError(
        413,
        "dataset_too_large",
        "This dataset is 1.2 GB; the current upload limit is 200 MB.",
      ),
    );
    await user.click(screen.getByRole("button", { name: "Upload and validate" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("dataset_too_large");
    expect(alert).toHaveTextContent("the current upload limit is 200 MB");
    expect(push).not.toHaveBeenCalled();
  });

  it("survives an unreachable server without losing the refusal", async () => {
    const user = userEvent.setup();
    render(<UploadForm />);
    await chooseFile(user);
    uploadMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await user.click(screen.getByRole("button", { name: "Upload and validate" }));
    expect(screen.getByRole("alert")).toHaveTextContent("network_error");
  });

  it("disables the action while the upload is in flight", async () => {
    const user = userEvent.setup();
    render(<UploadForm />);
    await chooseFile(user);
    let resolve!: (v: unknown) => void;
    uploadMock.mockReturnValueOnce(
      new Promise((res) => {
        resolve = res;
      }),
    );
    await user.click(screen.getByRole("button", { name: "Upload and validate" }));
    expect(screen.getByRole("button", { name: /validating/i })).toBeDisabled();
    resolve({});
  });
});
