import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Ask your inbox · DJ Inbox Assistant demo",
  description: "Ask a DJ company's inquiry inbox questions; every answer links to the emails it came from.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
