<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100">
    <div class="mx-auto max-w-5xl px-4 sm:px-6 py-6 sm:py-10">
      <div class="mb-6 sm:mb-8 text-center sm:text-left">
        <div class="flex flex-col items-center gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div class="flex items-center justify-center sm:justify-start gap-3">
            <img src="/logo.svg?v=ball4" alt="Logo" class="w-10 h-10 sm:w-12 sm:h-12 drop-shadow-[0_0_15px_rgba(239,68,68,0.5)]" />
            <div class="text-2xl sm:text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-violet-500 to-fuchsia-500">
              魔力彩票助手
            </div>
          </div>

          <div class="flex flex-col items-center sm:items-end">
            <div class="flex items-center gap-2 text-[11px] text-slate-400">
              <span class="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_14px_rgba(52,211,153,0.55)]"></span>
              <span>北京时间</span>
            </div>
            <div class="mt-1 text-2xl sm:text-[28px] leading-none font-extrabold text-slate-100 tabular-nums tracking-wide">
              {{ timeText }}
            </div>
            <div class="mt-1 text-xs text-slate-400 tabular-nums">
              {{ dateText }}
            </div>
          </div>
        </div>
        <div class="mt-2 text-xs text-slate-400">提示：彩票具有随机性，内容仅供娱乐与研究参考 <!-- v2.0.1 --></div>
      </div>

      <div class="rounded-2xl bg-white/5 backdrop-blur border border-white/10 shadow-xl">
        <div class="flex flex-wrap justify-center gap-2 p-3 sm:p-4 border-b border-white/10">
          <button
            v-for="tab in tabs"
            :key="tab.id"
            @click="currentTab = tab.id"
            :class="[
              'px-6 py-2.5 rounded-xl text-sm sm:text-base font-extrabold transition select-none',
              currentTab === tab.id
                ? 'bg-white text-slate-900'
                : 'bg-white/5 hover:bg-white/10 text-slate-100'
            ]"
          >
            {{ tab.name }}
          </button>
        </div>

        <div class="p-4 sm:p-6">
          <LotteryView :type="currentTab" />
        </div>
      </div>
    </div>
    
    <Toast ref="toastRef" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import LotteryView from './components/LotteryView.vue'
import Toast from './components/Toast.vue'
import { setToastInstance } from './composables/useToast.js'

const currentTab = ref('ssq')
const toastRef = ref(null)
const now = ref(new Date())
let nowTimer = null

const tabs = [
  { id: 'ssq', name: '双色球' },
  { id: 'dlt', name: '大乐透' }
]

const dateText = computed(() => {
  const d = now.value
  return d.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    weekday: 'short'
  })
})

const timeText = computed(() => {
  const d = now.value
  return d.toLocaleTimeString('zh-CN', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
})

onMounted(() => {
  if (toastRef.value) {
    setToastInstance(toastRef.value)
  }

  nowTimer = window.setInterval(() => {
    now.value = new Date()
  }, 1000)
})

onBeforeUnmount(() => {
  if (nowTimer) {
    window.clearInterval(nowTimer)
    nowTimer = null
  }
})
</script>
