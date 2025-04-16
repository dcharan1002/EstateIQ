"use client";

import { UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  const navItems = [
    { name: "Property Valuation", href: "/valuation", icon: "🏠" },
    { name: "Past Valuations", href: "/history", icon: "📅" },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 font-sans">
      {/* Navigation */}
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 justify-between items-center">
            <div className="flex items-center gap-8">
              {/* Logo */}
              <Link href="/" className="flex items-center gap-2">
                <span className="text-2xl font-black bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                  EstateIQ
                </span>
              </Link>
              
              {/* Navigation Links */}
              <div className="hidden sm:flex sm:space-x-8">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      pathname === item.href
                        ? "bg-blue-50 dark:bg-blue-900 text-blue-600 dark:text-blue-400"
                        : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                    }`}
                  >
                    <span>{item.icon}</span>
                    {item.name}
                  </Link>
                ))}
              </div>
            </div>

            {/* User Menu */}
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <UserButton 
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: "w-8 h-8",
                    userButtonPopoverCard: "bg-white dark:bg-gray-800 shadow-lg border border-gray-100 dark:border-gray-700",
                    userButtonPopoverText: "text-gray-700 dark:text-gray-300",
                    userButtonPopoverActionButton: "hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300",
                    userButtonPopoverActionButtonText: "text-gray-700 dark:text-gray-300",
                    userButtonPopoverFooter: "hidden",
                  }
                }}
              />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}
