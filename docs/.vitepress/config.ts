import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'MaiLLM',
  description: '基于 Qwen3.5 与 Gemma4 的端侧多模态小模型训练与蒸馏',
  base: '/MaiLLM/',
  head: [
    ['link', { rel: 'icon', href: '/MaiLLM/favicon.svg' }],
  ],
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '指南', link: '/guide/introduction' },
      { text: '训练方案研究', link: '/guide/research' },
    ],
    sidebar: {
      '/guide/': [
        {
          text: '入门',
          items: [
            { text: '项目定位与目标', link: '/guide/introduction' },
            { text: '训练流水线', link: '/guide/pipeline' },
            { text: '底座与教师选型', link: '/guide/models' },
            { text: '快速开始', link: '/guide/quickstart' },
          ],
        },
        {
          text: '深入',
          items: [
            { text: '训练详解', link: '/guide/training' },
            { text: '基准评测', link: '/guide/benchmark' },
            { text: '端侧部署', link: '/guide/deploy' },
            { text: '常见问题与风险', link: '/guide/faq' },
          ],
        },
        {
          text: '参考',
          items: [
            { text: '端侧模型训练方案研究', link: '/guide/research' },
          ],
        },
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Maicarons/MaiLLM' },
    ],
    footer: {
      message: 'MaiLLM · 端侧多模态小模型训练与蒸馏',
      copyright: 'Copyright © 2026 Mai',
    },
    docFooter: {
      prev: true,
      next: true,
    },
    lastUpdated: {
      text: '最后更新',
    },
    search: {
      provider: 'local',
    },
  },
})
