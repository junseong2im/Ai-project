'use client'

import { Gauge, Clock, Radio, Target, Award, Beaker, ShoppingBag, Zap } from 'lucide-react'

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

interface ResultsDisplayProps {
  results: ScentResult | null
  isLoading: boolean
}

export default function ResultsDisplay({ results, isLoading }: ResultsDisplayProps) {
  if (isLoading) {
    return (
      <div className="glass-morphism p-12 rounded-3xl max-w-4xl mx-auto text-center">
        <div className="loading-dots mx-auto mb-8">
          <div></div>
          <div></div>
          <div></div>
          <div></div>
        </div>
        <h3 className="text-2xl font-bold text-white mb-4">AI가 향수를 구현 중입니다</h3>
        <p className="text-white/80">
          5,757,842개 파라미터 신경망이 최적의 향조합을 계산하고 있습니다...
        </p>
        <div className="mt-8 space-y-2 text-white/60 text-sm">
          <div>✓ 장면 감정 분석 완료</div>
          <div>✓ 화학적 구조 매칭 중...</div>
          <div>⏳ 브랜드 제품 검색 중...</div>
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="glass-morphism p-12 rounded-3xl max-w-4xl mx-auto text-center">
        <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-white/10 flex items-center justify-center">
          <Beaker className="w-12 h-12 text-white/60" />
        </div>
        <h3 className="text-2xl font-bold text-white mb-4">분석 대기 중</h3>
        <p className="text-white/80">
          장면 분석 탭에서 영화 장면을 입력해주세요.
        </p>
      </div>
    )
  }

  const getIntensityColor = (value: number) => {
    if (value <= 3) return 'from-green-400 to-green-600'
    if (value <= 6) return 'from-yellow-400 to-orange-500'
    return 'from-orange-500 to-red-600'
  }

  const getMatchColor = (score: number) => {
    if (score >= 90) return 'text-green-400'
    if (score >= 80) return 'text-yellow-400'
    return 'text-orange-400'
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header Stats */}
      <div className="glass-morphism p-8 rounded-3xl">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold text-white">분석 결과</h2>
          <div className="flex items-center text-white/80">
            <Zap className="w-5 h-5 mr-2" />
            <span className="text-sm">{results.processing_time.toFixed(3)}초 처리</span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { icon: Gauge, label: '강도', value: results.intensity, max: 10, color: getIntensityColor(results.intensity) },
            { icon: Clock, label: '지속성', value: results.longevity, max: 15, color: 'from-blue-400 to-blue-600' },
            { icon: Radio, label: '투사력', value: results.projection, max: 10, color: 'from-purple-400 to-purple-600' },
            { icon: Target, label: '신뢰도', value: results.confidence, max: 100, color: 'from-emerald-400 to-emerald-600', suffix: '%' }
          ].map(({ icon: Icon, label, value, max, color, suffix = '' }) => (
            <div key={label} className="text-center">
              <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-white/10 flex items-center justify-center">
                <Icon className="w-8 h-8 text-white" />
              </div>
              <div className="text-2xl font-bold text-white mb-1">
                {value.toFixed(1)}{suffix}
              </div>
              <div className="text-white/80 text-sm mb-2">{label}</div>
              <div className="w-full bg-white/20 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full bg-gradient-to-r ${color}`}
                  style={{ width: `${(value / max) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Chemical Profile */}
        <div className="glass-morphism p-8 rounded-3xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <Beaker className="w-6 h-6 mr-3" />
            화학적 구조
          </h3>
          
          <div className="space-y-6">
            {[
              { label: 'Top Notes', notes: results.chemical_profile.top_notes, color: 'from-yellow-400 to-orange-400' },
              { label: 'Middle Notes', notes: results.chemical_profile.middle_notes, color: 'from-pink-400 to-rose-400' },
              { label: 'Base Notes', notes: results.chemical_profile.base_notes, color: 'from-amber-600 to-brown-600' }
            ].map(({ label, notes, color }) => (
              <div key={label}>
                <h4 className="text-white font-semibold mb-3">{label}</h4>
                <div className="flex flex-wrap gap-2">
                  {notes.map((note) => (
                    <span 
                      key={note}
                      className={`px-3 py-1 rounded-full text-white text-sm bg-gradient-to-r ${color} bg-opacity-80`}
                    >
                      {note}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Categories */}
        <div className="glass-morphism p-8 rounded-3xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
            <Award className="w-6 h-6 mr-3" />
            향수 카테고리
          </h3>
          
          <div className="grid grid-cols-1 gap-4">
            {results.category.map((category, index) => (
              <div key={category} className="scent-card p-4 rounded-2xl bg-gradient-to-r from-white/10 to-white/5 border border-white/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      index === 0 ? 'bg-yellow-400' : 
                      index === 1 ? 'bg-green-400' : 'bg-blue-400'
                    }`} />
                    <span className="text-white font-medium text-lg">{category}</span>
                  </div>
                  <div className="text-white/60 text-sm">
                    {index === 0 ? 'Primary' : index === 1 ? 'Secondary' : 'Accent'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="glass-morphism p-8 rounded-3xl">
        <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
          <ShoppingBag className="w-6 h-6 mr-3" />
          추천 제품
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.recommendations.map((product) => (
            <div key={`${product.brand}-${product.name}`} className="scent-card p-6 rounded-2xl bg-gradient-to-br from-white/15 to-white/5 border border-white/20 hover:border-white/30">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="text-white font-bold text-lg">{product.brand}</h4>
                  <p className="text-white/90 font-medium">{product.name}</p>
                </div>
                <div className={`text-right ${getMatchColor(product.match_score)}`}>
                  <div className="text-2xl font-bold">{product.match_score}</div>
                  <div className="text-xs">Match</div>
                </div>
              </div>
              
              <p className="text-white/70 text-sm mb-4 leading-relaxed">
                {product.description}
              </p>
              
              <div className="w-full bg-white/20 rounded-full h-2">
                <div 
                  className="h-2 rounded-full bg-gradient-to-r from-emerald-400 to-blue-500"
                  style={{ width: `${product.match_score}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Analysis Summary */}
      <div className="glass-morphism p-6 rounded-2xl bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/30">
        <div className="text-center">
          <p className="text-white/90 mb-2">
            <strong>AI 분석 완료:</strong> 신뢰도 {results.confidence.toFixed(1)}%로 
            총 {results.recommendations.length}개 제품을 추천합니다.
          </p>
          <p className="text-white/70 text-sm">
            화학적 구조 분석과 15개 카테고리 매칭을 통해 최적화된 결과입니다.
          </p>
        </div>
      </div>
    </div>
  )
}