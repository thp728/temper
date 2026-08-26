import type { Metadata } from "next";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import UploadForm from "@/components/UploadForm";

export const metadata: Metadata = {
  title: "Upload a dataset",
};

export default function UploadPage() {
  return (
    <section aria-labelledby="upload-heading" className="space-y-6">
      <div>
        <h1 id="upload-heading" className="text-2xl font-semibold">
          Upload a dataset
        </h1>
        <p className="mt-2 text-muted-foreground">
          Temper fine-tunes a base model on chat-format JSONL. Upload a file
          and its validation report comes back before anything is spent: every
          problem is named against the line it sits on.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Dataset</CardTitle>
          <CardDescription>
            The file is validated on upload; nothing runs until you launch.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <UploadForm />
        </CardContent>
      </Card>
    </section>
  );
}
