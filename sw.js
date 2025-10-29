
const CACHE_NAME = 'built-by-sadeem-cache-v1';
// This list includes the main app shell, scripts, and key external resources.
const urlsToCache = [
  '/',
  '/index.html',
  '/index.tsx',
  '/App.tsx',
  '/types.ts',
  '/services/gemini.ts',
  '/utils/fileHelper.ts',
  '/components/Modal.tsx',
  '/metadata.json',
  '/manifest.json',
  // Key CDNs
  'https://cdn.tailwindcss.com',
  'https://esm.sh/lucide-react@^0.525.0',
  'https://esm.sh/@google/genai@^1.11.0',
  'https://esm.sh/react@^19.1.0',
  'https://esm.sh/react-dom@^19.1.0',
  'https://esm.sh/mammoth@^1.9.1',
  'https://esm.sh/file-saver@^2.0.5',
  'https://esm.sh/jszip@^3.10.1'
];

// Install event: open a cache and add all assets to it
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
      .catch(err => {
          console.error('Failed to cache resources:', err);
      })
  );
});

// Activate event: clean up old caches
self.addEventListener('activate', (event) => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Fetch event: serve assets from cache first (cache-first strategy)
self.addEventListener('fetch', (event) => {
  // We only want to cache GET requests.
  if (event.request.method !== 'GET') {
    return;
  }
  
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // If the resource is in the cache, return it.
        if (response) {
          return response;
        }

        // Otherwise, fetch from the network.
        return fetch(event.request);
      })
  );
});