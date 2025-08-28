'use client'

import { useState } from 'react'
import { Play, Sparkles, MapPin, Clock, Gauge } from 'lucide-react'

interface SceneInputFormProps {
  onSubmit: (data: {
    description: string
    scene_type: string
    location?: string
    time_period?: string
    intensity: number
  }) => void
}

export default function SceneInputForm({ onSubmit }: SceneInputFormProps) {
  const [description, setDescription] = useState('')
  const [sceneType, setSceneType] = useState('romantic')
  const [location, setLocation] = useState('')
  const [timePeriod, setTimePeriod] = useState('')
  const [intensity, setIntensity] = useState(6)

  const sceneTypes = [
    { value: 'romantic', label: '로맨틱', emoji: '💕' },
    { value: 'horror', label: '공포', emoji: '👻' },
    { value: 'action', label: '액션', emoji: '💥' },
    { value: 'drama', label: '드라마', emoji: '🎭' },
    { value: 'comedy', label: '코미디', emoji: '😄' },
    { value: 'scifi', label: 'SF', emoji: '🚀' },
    { value: 'fantasy', label: '판타지', emoji: '🧙‍♂️' }
  ]

  const timeOptions = [
    { value: 'dawn', label: '새벽' },
    { value: 'morning', label: '아침' },
    { value: 'afternoon', label: '오후' },
    { value: 'evening', label: '저녁' },
    { value: 'night', label: '밤' }
  ]

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!description.trim()) return

    onSubmit({
      description: description.trim(),
      scene_type: sceneType,
      location: location.trim() || undefined,
      time_period: timePeriod || undefined,
      intensity
    })
  }

  const exampleScenes = [
    {
      type: 'romantic',
      text: '해변에서 석양을 바라보며 와인을 마시는 데이트'
    },
    {
      type: 'horror', 
      text: '어두운 지하실에서 괴물과 마주치는 순간'
    },
    {
      type: 'action',
      text: '빌딩 옥상에서 벌어지는 추격전'
    }
  ]

  const handleExampleClick = (example: { type: string; text: string }) => {
    setDescription(example.text)
    setSceneType(example.type)
  }

  return (
    <div className="glass-morphism p-8 rounded-3xl max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">영화 장면 분석</h2>
        <p className="text-white/80">
          원하는 영화 장면을 상세히 설명해주세요. AI가 완벽한 향수를 구현해드립니다.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Scene Description */}
        <div>
          <label className="block text-white font-semibold mb-4 text-lg">
            <Sparkles className="w-5 h-5 inline mr-2" />
            장면 설명
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="예: 비 오는 밤 옥상에서 이별하는 장면, 담배냄새와 빗물냄새가 섞인 쓸쓸한 분위기"
            className="w-full h-32 p-4 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/30 resize-none"
            required
          />
          <div className="text-right text-white/60 text-sm mt-2">
            {description.length}/500
          </div>
        </div>

        {/* Example Scenes */}
        <div>
          <label className="block text-white font-semibold mb-4">예시 장면</label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {exampleScenes.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleExampleClick(example)}
                className="p-4 rounded-xl bg-white/5 border border-white/20 text-white/80 hover:bg-white/10 hover:border-white/30 transition-all text-left text-sm"
              >
                {example.text}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Scene Type */}
          <div>
            <label className="block text-white font-semibold mb-4">
              <Play className="w-5 h-5 inline mr-2" />
              장면 타입
            </label>
            <div className="grid grid-cols-2 gap-2">
              {sceneTypes.map(({ value, label, emoji }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setSceneType(value)}
                  className={`p-3 rounded-xl border transition-all ${
                    sceneType === value
                      ? 'bg-white text-purple-600 border-white'
                      : 'bg-white/10 text-white border-white/20 hover:bg-white/20'
                  }`}
                >
                  <span className="text-lg mr-2">{emoji}</span>
                  <span className="font-medium">{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Time Period */}
          <div>
            <label className="block text-white font-semibold mb-4">
              <Clock className="w-5 h-5 inline mr-2" />
              시간대 (선택사항)
            </label>
            <select
              value={timePeriod}
              onChange={(e) => setTimePeriod(e.target.value)}
              className="w-full p-3 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-white/30"
            >
              <option value="">자동 감지</option>
              {timeOptions.map(({ value, label }) => (
                <option key={value} value={value} className="bg-purple-900">
                  {label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Location */}
        <div>
          <label className="block text-white font-semibold mb-4">
            <MapPin className="w-5 h-5 inline mr-2" />
            위치 (선택사항)
          </label>
          <input
            type="text"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="예: 옥상, 해변, 숲속, 지하실..."
            className="w-full p-3 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/30"
          />
        </div>

        {/* Intensity Slider */}
        <div>
          <label className="block text-white font-semibold mb-4">
            <Gauge className="w-5 h-5 inline mr-2" />
            향수 강도: {intensity}/10
          </label>
          <div className="relative">
            <input
              type="range"
              min="1"
              max="10"
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-white/60 text-sm mt-2">
              <span>은은함</span>
              <span>적당함</span>
              <span>강렬함</span>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!description.trim()}
          className="w-full py-4 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-bold text-lg rounded-2xl transition-all transform hover:scale-[1.02] active:scale-[0.98] pulse-glow"
        >
          <Play className="w-6 h-6 inline mr-3" />
          향수 구현하기
        </button>
      </form>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 25px;
          width: 25px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          cursor: pointer;
          box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        }

        .slider::-moz-range-thumb {
          height: 25px;
          width: 25px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        }
      `}</style>
    </div>
  )
}