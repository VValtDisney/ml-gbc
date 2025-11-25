export default {
    esbuild: {
        drop: ['console', 'debugger'],
    },
    server: {
        host: '0.0.0.0',
        port: process.env.VITE_PORT || 5173,
        proxy: {
            '/api': {
                // Replace target with your API server address
                // Only meaningful for dev server
                target: 'http://localhost:5050',
                changeOrigin: true
            }
        }
    }
}