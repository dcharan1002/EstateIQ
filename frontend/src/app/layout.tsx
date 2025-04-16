import { type Metadata } from 'next'
import { ClerkProvider } from '@clerk/nextjs'
import { Providers } from './providers'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'EstateIQ - AI-Powered Property Valuation',
  description: 'Get instant property valuations using advanced AI technology',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>
          <ClerkProvider>{children}</ClerkProvider>
        </Providers>
      </body>
    </html>
  )
}
