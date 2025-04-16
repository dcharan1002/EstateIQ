"use client";

import { SignUp } from "@clerk/nextjs";
import { dark } from "@clerk/themes";
import { useTheme } from "next-themes";
import Link from "next/link";

export default function SignUpPage() {
  const { resolvedTheme } = useTheme();
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 relative font-sans">
      <div
        className="absolute inset-0 z-0 opacity-50"
        style={{
          backgroundImage: "url('/images/unslash.jpg')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundAttachment: "fixed",
        }}
      />
      
      <div className="relative z-10 w-full max-w-md">
        <div className="mb-8 text-center">
          <Link href="/" className="inline-block">
            <h1 className="text-4xl font-black bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
              EstateIQ
            </h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">Intelligent Real Estate Valuation</p>
          </Link>
        </div>

        <div className="p-8 rounded-2xl">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6 text-center">Create Account</h2>
          <SignUp 
            routing="path"
            path="/sign-up"
            redirectUrl="/valuation"
            appearance={{
              baseTheme: resolvedTheme === "dark" ? dark : undefined,
              elements: {
                card: "bg-transparent shadow-none",
                headerTitle: "hidden",
                headerSubtitle: "text-gray-600 dark:text-gray-400",
                socialButtonsBlockButton: "border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800",
                socialButtonsBlockButtonText: "text-gray-700 dark:text-gray-300",
                formButtonPrimary: "bg-blue-500 hover:bg-blue-600",
                formFieldLabel: "text-gray-700 dark:text-gray-300",
                formFieldInput: "bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-900 dark:text-white",
                footerActionText: "text-gray-600 dark:text-gray-400",
                formFieldError: "text-red-600 dark:text-red-400",
                footerActionLink: "text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300",
              },
            }}
          />
        </div>
      </div>
    </main>
  );
}
