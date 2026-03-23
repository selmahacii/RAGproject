import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Selma Search Hub",
  description: "Modern search hub for high-performance data processing. Built with TypeScript, Tailwind CSS, and shadcn/ui.",
  keywords: ["Selma", "Haci", "Search Hub", "Data", "Efficiency"],
  authors: [{ name: "Selma Haci" }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: "Selma Search Hub",
    description: "Empowering business with modern search solutions",
    url: "https://selma.haci.com",
    siteName: "Selma Search Hub",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Selma Search Hub",
    description: "Empowering business with modern search solutions",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
