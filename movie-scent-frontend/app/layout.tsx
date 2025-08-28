import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Movie Scent AI - 영화 향수 AI 시스템',
  description: '감독이 원하는 어떤 향이든 AI가 구현해드립니다. 영화 장면을 향수로 변환하는 혁신적인 딥러닝 시스템.',
  keywords: ['AI', '향수', '영화', '딥러닝', '감독', '냄새', '향료'],
  authors: [{ name: 'Movie Scent AI Team' }],
  openGraph: {
    title: 'Movie Scent AI - 영화 향수 AI 시스템',
    description: '감독이 원하는 어떤 향이든 AI가 구현해드립니다',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Movie Scent AI - 영화 향수 AI 시스템',
    description: '감독이 원하는 어떤 향이든 AI가 구현해드립니다',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body>
        <div className="min-h-screen bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-800">
          {children}
        </div>
      </body>
    </html>
  )
}