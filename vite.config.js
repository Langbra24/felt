import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  optimizeDeps: {
    include: ['p5', 'tweakpane'],
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        boids: resolve(__dirname, 'experiments/01-boids/index.html'),
      }
    }
  }
})
