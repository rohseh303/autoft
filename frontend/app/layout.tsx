import "./globals.css";

export const metadata = {
  title: "AutoFT — autonomous fine-tuning",
  description: "Paste a task, get a fine-tuned open-source LLM.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
