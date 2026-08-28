// A held-out loss that stopped improving, said in plain language (issue #53).
// One rendering for both surfaces: the running view announces it as a live
// region (role="status") and the finished record shows it as a quiet note.

export default function PlateauNote({
  message,
  live = false,
}: {
  message: string;
  live?: boolean;
}) {
  const role = live ? { role: "status" as const } : {};
  return (
    <p
      {...role}
      className="rounded-lg border bg-muted/50 p-3 text-sm"
    >
      {message}
    </p>
  );
}
