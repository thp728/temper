import Link from "next/link";
import { Button } from "@/components/ui/button";

// One way back to the start of the journey, wherever a dead end puts you.
export default function BackToUpload() {
  return (
    <Button variant="outline" asChild>
      <Link href="/datasets">Back to upload</Link>
    </Button>
  );
}
