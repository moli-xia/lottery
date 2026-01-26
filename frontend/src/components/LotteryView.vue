<template>
  <div class="space-y-6">
    <!-- Latest Result Highlight -->
    <div v-if="latestResult" class="rounded-2xl border border-amber-500/30 bg-gradient-to-r from-amber-500/10 to-orange-500/10 p-6 shadow-[0_0_20px_rgba(251,191,36,0.1)]">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div class="flex flex-col justify-center">
          <div class="flex items-center gap-3">
            <span class="px-3 py-1 rounded-full bg-amber-500/20 text-amber-300 text-xs font-bold border border-amber-500/30 inline-flex items-center h-7">最新开奖</span>
            <h2 class="text-2xl font-extrabold text-white leading-none">第 {{ latestResult.issue }} 期</h2>
          </div>
          <div class="mt-2 text-slate-300 text-sm">{{ latestResult.date }}</div>
        </div>
        <div class="flex flex-wrap gap-3">
          <span
            v-for="n in splitNums(latestResult.red_balls)"
            :key="`lr-${n}`"
            class="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-gradient-to-br from-rose-500 to-red-600 text-white flex items-center justify-center text-2xl sm:text-3xl font-black shadow-lg shadow-rose-900/50"
          >{{ n }}</span>
          <span
            v-for="n in splitNums(latestResult.blue_balls)"
            :key="`lb-${n}`"
            class="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-gradient-to-br from-sky-500 to-blue-600 text-white flex items-center justify-center text-2xl sm:text-3xl font-black shadow-lg shadow-sky-900/50"
          >{{ n }}</span>
        </div>
      </div>
    </div>

    <div class="rounded-2xl border border-white/10 bg-gradient-to-br from-violet-500/20 via-white/5 to-sky-500/15 p-5">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div class="space-y-2">
          <div class="flex items-center gap-3">
            <h2 class="text-xl font-extrabold">AI 推算</h2>
          </div>
          <div class="text-sm text-slate-300">
            开奖后自动生成多组号码组合，点击可继续追加
            <span class="ml-2 text-[11px] text-slate-400">{{ llmNotice }}</span>
          </div>
        </div>
      </div>

      <div v-if="predictions.length > 0" class="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div
          v-for="(p, idx) in predictions"
          :key="p.id || `p-${idx}`"
          class="rounded-2xl border border-white/10 bg-white/5 p-4"
        >
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-extrabold text-slate-100">第 {{ idx + 1 }} 组</div>
            <div v-if="p.based_on_issue" class="text-xs text-slate-400">基于近100期（截至第 {{ p.based_on_issue }} 期）</div>
          </div>
          <div class="mt-3 flex flex-wrap gap-2">
            <span
              v-for="n in splitNums(p.red_balls)"
              :key="`pr-${p.id}-${n}`"
              class="w-10 h-10 sm:w-11 sm:h-11 rounded-full bg-rose-500/90 text-white flex items-center justify-center text-base sm:text-lg font-extrabold shadow-[0_0_0_5px_rgba(244,63,94,0.16)]"
            >{{ n }}</span>
            <span
              v-for="n in splitNums(p.blue_balls)"
              :key="`pb-${p.id}-${n}`"
              class="w-10 h-10 sm:w-11 sm:h-11 rounded-full bg-sky-500/90 text-white flex items-center justify-center text-base sm:text-lg font-extrabold shadow-[0_0_0_5px_rgba(14,165,233,0.16)]"
            >{{ n }}</span>
          </div>
        </div>
      </div>

      <div v-else class="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 text-slate-300">
        {{ predictionsLoading ? '生成中…' : '暂无推算结果，等待自动抓取与生成' }}
      </div>
      <div v-if="predictionsError" class="text-rose-300 mt-3">{{ predictionsError }}</div>
      <button
        v-if="predictions.length > 0"
        type="button"
        @click="loadMorePredictions"
        :disabled="predictionsLoading"
        class="mt-4 mx-auto w-9 h-9 rounded-full bg-white/10 hover:bg-white/15 border border-white/10 text-slate-100 flex items-center justify-center disabled:opacity-50"
        aria-label="加载更多推算"
        title="加载更多推算"
      >
        <svg
          v-if="predictionsLoading"
          viewBox="0 0 24 24"
          fill="none"
          class="w-4 h-4 animate-spin"
        >
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"></path>
        </svg>
        <svg v-else viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
          <path fill-rule="evenodd" d="M10 13.5a1 1 0 0 1-.707-.293l-4-4a1 1 0 1 1 1.414-1.414L10 11.086l3.293-3.293a1 1 0 1 1 1.414 1.414l-4 4A1 1 0 0 1 10 13.5z" clip-rule="evenodd" />
        </svg>
      </button>
    </div>

    <!-- Statistics & Charts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6" v-if="history.length > 0">
      <!-- Red Ball Hit Rate -->
      <div class="rounded-2xl border border-white/10 bg-white/5 p-5">
        <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
          <span class="w-1 h-5 bg-rose-500 rounded-full"></span>
          红球命中率排行 (近100期)
        </h3>
        <div class="space-y-3">
          <div v-for="(item, idx) in redBallStats.slice(0, 8)" :key="`hot-r-${item.num}`" class="flex items-center gap-3">
            <div class="w-6 text-sm text-slate-400 font-mono">#{{ idx + 1 }}</div>
            <div class="w-8 h-8 rounded-full bg-rose-500/20 text-rose-300 flex items-center justify-center text-sm font-bold border border-rose-500/30">{{ item.num }}</div>
            <div class="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
              <div class="h-full bg-rose-500" :style="{ width: `${item.rate * 100}%` }"></div>
            </div>
            <div class="w-24 text-right text-sm text-slate-300 tabular-nums">{{ (item.rate * 100).toFixed(1) }}% · {{ item.count }}期</div>
          </div>
        </div>
      </div>

      <!-- Blue Ball Hit Rate -->
      <div class="rounded-2xl border border-white/10 bg-white/5 p-5">
        <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
          <span class="w-1 h-5 bg-sky-500 rounded-full"></span>
          蓝球命中率排行 (近100期)
        </h3>
        <div class="space-y-3">
          <div v-for="(item, idx) in blueBallStats.slice(0, 8)" :key="`hot-b-${item.num}`" class="flex items-center gap-3">
            <div class="w-6 text-sm text-slate-400 font-mono">#{{ idx + 1 }}</div>
            <div class="w-8 h-8 rounded-full bg-sky-500/20 text-sky-300 flex items-center justify-center text-sm font-bold border border-sky-500/30">{{ item.num }}</div>
            <div class="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
              <div class="h-full bg-sky-500" :style="{ width: `${item.rate * 100}%` }"></div>
            </div>
            <div class="w-24 text-right text-sm text-slate-300 tabular-nums">{{ (item.rate * 100).toFixed(1) }}% · {{ item.count }}期</div>
          </div>
        </div>
      </div>
    </div>

    <!-- History List -->
    <div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between pt-4 border-t border-white/10">
      <div class="text-center sm:text-left">
        <h2 class="text-xl font-bold">{{ title }} 历史记录 (近100期)</h2>
        <div class="mt-1 text-xs text-slate-400">开奖后自动抓取，抓到新数据后更新一次</div>
      </div>
    </div>

    <div class="sm:hidden space-y-3">
      <div
        v-for="item in historyList"
        :key="`m-${item.issue}`"
        class="rounded-2xl border border-white/10 bg-white/5 p-4"
      >
        <div class="flex items-center justify-between gap-3">
          <div class="font-extrabold text-slate-100">第 {{ item.issue }} 期</div>
          <div class="text-xs text-slate-300">{{ item.date }}</div>
        </div>
        <div class="mt-3 flex flex-wrap items-center gap-2">
          <div class="flex flex-wrap gap-2">
            <span
              v-for="n in splitNums(item.red_balls)"
              :key="`${item.issue}-mr-${n}`"
              class="w-9 h-9 rounded-full bg-rose-500/90 text-white text-base font-extrabold flex items-center justify-center"
            >
              {{ n }}
            </span>
          </div>
          <div class="w-px h-7 bg-white/10"></div>
          <div class="flex flex-wrap gap-2">
            <span
              v-for="n in splitNums(item.blue_balls)"
              :key="`${item.issue}-mb-${n}`"
              class="w-9 h-9 rounded-full bg-sky-500/90 text-white text-base font-extrabold flex items-center justify-center"
            >
              {{ n }}
            </span>
          </div>
        </div>
      </div>
      <div v-if="history.length === 0" class="rounded-2xl border border-white/10 bg-white/5 p-6 text-center text-slate-300">
        暂无数据，等待自动抓取
      </div>
    </div>

    <div class="hidden sm:block rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="min-w-full">
          <thead class="bg-white/5">
            <tr>
              <th class="px-3 sm:px-4 py-3 text-left text-xs font-semibold text-slate-300">期号</th>
              <th class="px-3 sm:px-4 py-3 text-left text-xs font-semibold text-slate-300">日期</th>
              <th class="px-3 sm:px-4 py-3 text-left text-xs font-semibold text-slate-300">开奖号码</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/10">
            <tr v-for="item in historyList" :key="item.issue" class="hover:bg-white/5">
              <td class="px-3 sm:px-4 py-3 whitespace-nowrap font-semibold text-slate-100">{{ item.issue }}</td>
              <td class="px-3 sm:px-4 py-3 whitespace-nowrap text-slate-200">{{ item.date }}</td>
              <td class="px-3 sm:px-4 py-3">
                <div class="flex flex-wrap items-center gap-2">
                  <div class="flex flex-wrap gap-2">
                    <span
                      v-for="n in splitNums(item.red_balls)"
                      :key="`${item.issue}-r-${n}`"
                      class="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-rose-500/90 text-white text-xs sm:text-sm font-bold flex items-center justify-center"
                    >
                      {{ n }}
                    </span>
                  </div>
                  <div class="w-px h-6 bg-white/10"></div>
                  <div class="flex flex-wrap gap-2">
                    <span
                      v-for="n in splitNums(item.blue_balls)"
                      :key="`${item.issue}-b-${n}`"
                      class="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-sky-500/90 text-white text-xs sm:text-sm font-bold flex items-center justify-center"
                    >
                      {{ n }}
                    </span>
                  </div>
                </div>
              </td>
            </tr>
            <tr v-if="history.length === 0">
              <td class="px-4 py-10 text-center text-slate-300" colspan="3">暂无数据，等待自动抓取</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <Teleport to="body">
      <Transition
        enter-active-class="transition ease-out duration-300"
        enter-from-class="opacity-0 translate-y-3"
        enter-to-class="opacity-100 translate-y-0"
        leave-active-class="transition ease-in duration-200"
        leave-from-class="opacity-100 translate-y-0"
        leave-to-class="opacity-0 translate-y-3"
      >
        <button
          v-if="showToTopButton"
          type="button"
          @click="scrollToTop"
          class="fixed z-[100] w-11 h-11 rounded-full bg-slate-950/70 text-white shadow-xl shadow-slate-950/40 border border-white/10 backdrop-blur hover:bg-slate-950/80 active:scale-95 transition-all duration-200 flex items-center justify-center"
          style="right: calc(1rem + env(safe-area-inset-right)); bottom: calc(1rem + env(safe-area-inset-bottom));"
          aria-label="一键至顶"
          title="一键至顶"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5">
            <path fill-rule="evenodd" d="M10 16a1 1 0 0 1-1-1V6.414L6.707 8.707a1 1 0 1 1-1.414-1.414l4.004-4.004a1 1 0 0 1 1.414 0l4.004 4.004a1 1 0 1 1-1.414 1.414L11 6.414V15a1 1 0 0 1-1 1z" clip-rule="evenodd" />
          </svg>
        </button>
      </Transition>
    </Teleport>
  </div>
</template>


<script setup>
import { computed, ref, watch, onBeforeUnmount, onMounted } from 'vue'
import { useToast } from '../composables/useToast.js'

const toast = useToast()

const props = defineProps(['type'])
const history = ref([])
const loading = ref(false)
const predictionsLoading = ref(false)
const predictionsError = ref(null)
const predictions = ref([])
const llmNotice = ref('大模型状态：检测中…')
const latestIssue = ref(null)
let latestPollTimer = null
const showToTopButton = ref(false)
const hasUserScrolled = ref(false)
let latestStream = null
const HISTORY_LIMIT = 100
const DEFAULT_PREDICTIONS = 2
const PREFETCH_PREDICTIONS = 20

const historyWindow = computed(() => {
  return history.value.slice(0, HISTORY_LIMIT)
})

const title = computed(() => {
  if (props.type === 'ssq') return '双色球'
  if (props.type === 'dlt') return '大乐透'
  return props.type
})

const latestResult = computed(() => {
  return historyWindow.value.length > 0 ? historyWindow.value[0] : null
})

const historyList = computed(() => {
  return historyWindow.value
})

const redBallStats = computed(() => {
  const stats = {}
  historyWindow.value.forEach(item => {
    splitNums(item.red_balls).forEach(n => {
      stats[n] = (stats[n] || 0) + 1
    })
  })
  const denom = historyWindow.value.length || 1
  return Object.entries(stats)
    .map(([num, count]) => ({ num, count, rate: count / denom }))
    .sort((a, b) => (b.rate - a.rate) || (b.count - a.count) || (String(a.num).localeCompare(String(b.num))))
})

const blueBallStats = computed(() => {
  const stats = {}
  historyWindow.value.forEach(item => {
    splitNums(item.blue_balls).forEach(n => {
      stats[n] = (stats[n] || 0) + 1
    })
  })

  const denom = historyWindow.value.length || 1
  return Object.entries(stats)
    .map(([num, count]) => ({ num, count, rate: count / denom }))
    .sort((a, b) => (b.rate - a.rate) || (b.count - a.count) || (String(a.num).localeCompare(String(b.num))))
})

const splitNums = (value) => {
  if (!value) return []
  return String(value)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

const fetchHistory = async () => {
  loading.value = true
  try {
    const res = await fetch(`/api/history/${props.type}?limit=${HISTORY_LIMIT}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    history.value = await res.json()
  } catch (error) {
    console.error('Failed to fetch history:', error)
    toast.error('加载历史数据失败', error.message)
  } finally {
    loading.value = false
  }
}

const fetchPredictionItems = async (limit = 5, offset = 0) => {
  const res = await fetch(`/api/predictions/${props.type}?limit=${limit}&offset=${offset}`)
  const data = await res.json()
  const items = Array.isArray(data.items) ? data.items : []
  return { basedOnIssue: data.based_on_issue, items, llm: data.llm || null }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const ensurePredictionsTotal = async (minCount) => {
  const res = await fetch(`/api/ensure_predictions/${props.type}?min_count=${minCount}`, { method: 'POST' })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || '生成失败')
  }
}

const waitForPredictionItems = async (want, offset, maxWaitMs = 120000) => {
  const startedAt = Date.now()
  let last = null
  while (Date.now() - startedAt < maxWaitMs) {
    last = await fetchPredictionItems(want, offset)
    if (last.items.length >= want) return last
    await sleep(2500)
  }
  return last || { basedOnIssue: null, items: [], llm: null }
}

const setLlmNoticeFromResponse = (llm) => {
  if (!llm) {
    return
  }
  const hasSignal =
    Object.prototype.hasOwnProperty.call(llm, 'used') ||
    Object.prototype.hasOwnProperty.call(llm, 'used_llm') ||
    Boolean(llm.model) ||
    Number.isFinite(llm.latency_ms)
  if (!hasSignal) {
    return
  }
  if (llm.unknown) {
    if (llm.configured && llm.configured_model) {
      llmNotice.value = `大模型已配置（未执行） · ${llm.configured_model}`
      return
    }
    llmNotice.value = '尚未配置大模型'
    return
  }
  const used = llm.used === true || llm.used_llm === true
  const ms = Number.isFinite(llm.latency_ms) ? llm.latency_ms : null
  const sec = ms == null ? null : (ms / 1000).toFixed(1)
  const parts = [used ? '已执行大模型推算' : '未使用大模型推算']
  if (used && llm.model) parts.push(llm.model)
  if (used && sec != null) parts.push(`${sec}s`)
  llmNotice.value = parts.join(' · ')
}

const ensureDefaultPredictions = async () => {
  predictionsLoading.value = true
  predictionsError.value = null
  try {
    const r = await fetchPredictionItems(DEFAULT_PREDICTIONS, 0)
    if (!r.basedOnIssue) {
      predictions.value = []
      llmNotice.value = '暂无开奖数据'
      return
    }
    if (r.llm) setLlmNoticeFromResponse(r.llm)
    predictions.value = r.items
    if (!r.llm || r.llm.used !== true) {
      predictions.value = []
      if (r.llm && r.llm.unknown && r.llm.configured) {
        llmNotice.value = `大模型推算中… · ${r.llm.configured_model || ''}`.trim()
        ;(async () => {
          try {
            await ensurePredictionsTotal(PREFETCH_PREDICTIONS)
            const r3 = await waitForPredictionItems(DEFAULT_PREDICTIONS, 0)
            if (r3.llm) setLlmNoticeFromResponse(r3.llm)
            if (r3.llm && r3.llm.used === true) {
              predictions.value = r3.items
            } else {
              predictions.value = []
              llmNotice.value = '大模型调用失败，已隐藏推算结果'
            }
          } catch (e) {
            console.error('Generate predictions failed:', e)
          }
        })()
      } else if (r.llm && r.llm.used === false) {
        llmNotice.value = '大模型调用失败，已隐藏推算结果'
      } else {
        llmNotice.value = '大模型未执行，已隐藏推算结果'
      }
    }
    ;(async () => {
      try {
        await ensurePredictionsTotal(PREFETCH_PREDICTIONS)
      } catch (e) {
        console.error('Prefetch predictions failed:', e)
      }
    })()
  } catch (e) {
    predictionsError.value = e.message
    toast.error('加载推算失败', e.message)
  } finally {
    predictionsLoading.value = false
  }
}

const loadMorePredictions = async () => {
  predictionsLoading.value = true
  predictionsError.value = null
  try {
    if (!predictions.value || predictions.value.length === 0) {
      throw new Error('大模型未执行，已隐藏推算结果')
    }
    const want = 2
    const baseOffset = predictions.value.length
    const r = await fetchPredictionItems(want, baseOffset)
    let newItems = r.items
    if (newItems.length < want) {
      // 检查LLM状态，如果已经成功执行且items为空，说明已加载完所有推算
      if (r.llm && r.llm.used === true && newItems.length === 0) {
        llmNotice.value = '已加载全部推算结果'
        return // 不触发新推算
      }
      
      // 否则尝试生成新推算
      llmNotice.value = '大模型推算中…'
      await ensurePredictionsTotal(Math.min(30, baseOffset + want))
      const r2 = await waitForPredictionItems(want, baseOffset)
      if (r2.llm) setLlmNoticeFromResponse(r2.llm)
      newItems = r2.items
      if (!newItems.length) {
        llmNotice.value = '大模型调用失败，已隐藏推算结果'
      }
    }
    predictions.value = [...predictions.value, ...newItems]
  } catch (e) {
    predictionsError.value = e.message
  } finally {
    predictionsLoading.value = false
  }
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

const onScroll = () => {
  hasUserScrolled.value = true
  showToTopButton.value = hasUserScrolled.value && window.scrollY > 300
}

const startLatestStream = () => {
  try {
    if (latestStream) latestStream.close()
    latestStream = new EventSource(`/api/stream/latest/${props.type}`)

    latestStream.addEventListener('init', (evt) => {
      const issue = evt && evt.data ? String(evt.data) : null
      if (issue) latestIssue.value = issue
    })

    latestStream.addEventListener('update', async (evt) => {
      const issue = evt && evt.data ? String(evt.data) : null
      if (issue && issue !== latestIssue.value) {
        latestIssue.value = issue
        await fetchHistory()
        predictions.value = []
        await ensureDefaultPredictions()
      }
      if (latestStream) latestStream.close()
      latestStream = null
    })

    latestStream.addEventListener('timeout', () => {
      if (latestStream) latestStream.close()
      latestStream = null
    })

    latestStream.onerror = () => {
      if (latestStream) latestStream.close()
      latestStream = null
    }
  } catch (e) {
    predictionsError.value = (e && e.message) ? e.message : String(e)
  }
}

watch(() => props.type, () => {
  history.value = []
  predictions.value = []
  latestIssue.value = null
  fetchHistory().then(() => ensureDefaultPredictions())
  startLatestStream()
})

onMounted(() => {
  fetchHistory().then(() => ensureDefaultPredictions())
  startLatestStream()
  window.addEventListener('scroll', onScroll, { passive: true })
  showToTopButton.value = false
})

onBeforeUnmount(() => {
  if (latestPollTimer) window.clearInterval(latestPollTimer)
  latestPollTimer = null
  if (latestStream) latestStream.close()
  latestStream = null
  window.removeEventListener('scroll', onScroll)
})
</script>
