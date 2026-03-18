import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "../context/AuthContext";
import Navbar from "../components/Navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata = {
  title: "Hybrid Intelligence Portfolio System",
  description: "AI-Powered Portfolio Management with Market Adaptability",
  manifest: "/manifest.json",
};

export const viewport = {
  themeColor: "#0f111a",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.variable}>
        <AuthProvider>
          <div className="page-wrapper">
            <Navbar />
            <main className="main-content">
              {children}
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
