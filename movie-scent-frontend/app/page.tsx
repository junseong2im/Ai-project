'use client'

import { useState } from 'react'
import { Film, Sparkles, Brain, ChevronRight, BarChart3, Beaker, Zap } from 'lucide-react'
import SceneInputForm from './components/SceneInputForm'
import ResultsDisplay from './components/ResultsDisplay'
import TrainingVisualization from './components/TrainingVisualization'

interface ScentResult {
  intensity: number
  longevity: number
  projection: number
  confidence: number
  category: string[]
  recommendations: Array<{
    brand: string
    name: string
    match_score: number
    description: string
  }>
  chemical_profile: {
    top_notes: string[]
    middle_notes: string[]
    base_notes: string[]
  }
  processing_time: number
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<'input' | 'results' | 'training'>('input')
  const [results, setResults] = useState<ScentResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSceneAnalysis = async (sceneData: {
    description: string
    scene_type: string
    location?: string
    time_period?: string
    intensity: number
  }) => {
    setIsLoading(true)
    setActiveTab('results')
    
    try {
      // 시뮬레이션된 AI 분석 결과 (실제 환경에서는 API 호출)
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      const mockResult: ScentResult = {
        intensity: sceneData.intensity,
        longevity: Math.random() * 10 + 5,
        projection: Math.random() * 10 + 4,
        confidence: Math.random() * 20 + 80,
        category: generateCategories(sceneData.scene_type),
        recommendations: generateRecommendations(sceneData.scene_type),
        chemical_profile: generateChemicalProfile(sceneData.scene_type, sceneData.description),
        processing_time: Math.random() * 0.1 + 0.05
      }
      
      setResults(mockResult)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const generateCategories = (sceneType: string): string[] => {
    const categoryMap: { [key: string]: string[] } = {
      romantic: ['Floral', 'Oriental', 'Gourmand'],
      horror: ['Smoky', 'Earthy', 'Metallic'],
      action: ['Woody', 'Spicy', 'Fresh'],
      drama: ['Powdery', 'Musky', 'Soft'],
      comedy: ['Citrus', 'Fresh', 'Light'],
      scifi: ['Ozone', 'Metallic', 'Synthetic'],
      fantasy: ['Mystical', 'Herbal', 'Ethereal']
    }
    return categoryMap[sceneType] || ['Fresh', 'Clean', 'Modern']
  }

  const generateRecommendations = (sceneType: string) => {
    const recommendations: { [key: string]: any[] } = {
      romantic: [
        { brand: 'Chanel', name: 'Coco Mademoiselle', match_score: 94, description: 'Elegant and sophisticated with rose and jasmine' },
        { brand: 'Dior', name: "J'adore", match_score: 89, description: 'Luminous floral with ylang-ylang and Damascus rose' },
        { brand: 'Tom Ford', name: 'Black Orchid', match_score: 87, description: 'Luxurious and sensual with black truffle and orchid' }
      ],
      horror: [
        { brand: 'Tom Ford', name: 'Tobacco Vanille', match_score: 93, description: 'Dark and smoky with tobacco leaf and vanilla' },
        { brand: 'Maison Margiela', name: 'By the Fireplace', match_score: 90, description: 'Campfire smoke with chestnuts and vanilla' },
        { brand: 'Lalique', name: 'Encre Noire', match_score: 86, description: 'Dark vetiver with earthy and smoky facets' }
      ],
      action: [
        { brand: 'Creed', name: 'Aventus', match_score: 95, description: 'Bold and powerful with pineapple and oakmoss' },
        { brand: 'Bleu de Chanel', name: 'Parfum', match_score: 91, description: 'Intense and sophisticated with sandalwood' },
        { brand: 'Giorgio Armani', name: 'Code', match_score: 88, description: 'Mysterious and seductive with bergamot and tonka' }
      ]
    }
    return recommendations[sceneType] || recommendations.romantic
  }

  const generateChemicalProfile = (sceneType: string, description: string) => {
    const profiles: { [key: string]: any } = {
      romantic: {
        top_notes: ['Rose Petals', 'Bergamot', 'Pink Pepper'],
        middle_notes: ['Jasmine', 'Orange Blossom', 'Lily of Valley'],
        base_notes: ['White Musk', 'Sandalwood', 'Vanilla']
      },
      horror: {
        top_notes: ['Black Pepper', 'Smoke', 'Metallic Notes'],
        middle_notes: ['Burnt Wood', 'Tar', 'Leather'],
        base_notes: ['Patchouli', 'Dark Amber', 'Vetiver']
      },
      action: {
        top_notes: ['Adrenaline Accord', 'Citrus', 'Mint'],
        middle_notes: ['Geranium', 'Lavender', 'Spices'],
        base_notes: ['Oakmoss', 'Cedarwood', 'Ambergris']
      }
    }
    return profiles[sceneType] || profiles.romantic
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <header className="text-center mb-12">
        <div className="flex items-center justify-center mb-6">
          <div className="glass-morphism p-4 rounded-full mr-4">
            <Film className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-white">
            Movie Scent AI
          </h1>
          <div className="glass-morphism p-4 rounded-full ml-4">
            <Sparkles className="w-12 h-12 text-white" />
          </div>
        </div>
        <p className="text-xl md:text-2xl text-white/90 mb-4">
          감독이 원하는 어떤 향이든 AI가 구현해드립니다
        </p>
        <div className="flex items-center justify-center text-white/80">
          <Brain className="w-5 h-5 mr-2" />
          <span className="text-sm">5,757,842개 파라미터 딥러닝 신경망</span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="glass-morphism p-2 rounded-2xl mb-8 max-w-2xl mx-auto">
        <div className="flex space-x-1">
          {[
            { id: 'input', icon: Beaker, label: '장면 분석' },
            { id: 'results', icon: ChevronRight, label: '분석 결과' },
            { id: 'training', icon: BarChart3, label: '학습 그래프' }
          ].map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex-1 flex items-center justify-center py-3 px-4 rounded-xl transition-all ${
                activeTab === id
                  ? 'bg-white text-purple-600 shadow-lg'
                  : 'text-white/70 hover:text-white hover:bg-white/10'
              }`}
            >
              <Icon className="w-5 h-5 mr-2" />
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-6xl mx-auto">
        {activeTab === 'input' && (
          <div className="animate-fade-in">
            <SceneInputForm onSubmit={handleSceneAnalysis} />
          </div>
        )}
        
        {activeTab === 'results' && (
          <div className="animate-fade-in">
            <ResultsDisplay results={results} isLoading={isLoading} />
          </div>
        )}
        
        {activeTab === 'training' && (
          <div className="animate-fade-in">
            <TrainingVisualization />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center mt-16 text-white/60">
        <div className="flex items-center justify-center mb-4">
          <Zap className="w-4 h-4 mr-2" />
          <span className="text-sm">실시간 처리 • 0.1초 내 응답 • 15개 향수 카테고리</span>
        </div>
        <p className="text-xs">
          Movie Scent AI System v1.0 | 딥러닝 기반 향수 추천 엔진
        </p>
      </footer>
    </div>
  )
}