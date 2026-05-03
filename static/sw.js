const CACHE_NAME = 'trusttag-v1';
const urlsToCache = [
  '/',
  '/static/manifest.json',
  'https://cdn.tailwindcss.com'
];

// Install event - cache resources
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // Clone the request
        const fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(
          response => {
            // Check if valid response
            if(!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then(cache => {
                // Cache successful responses for static assets
                if (event.request.url.includes('/static/') || 
                    event.request.url.includes('cdn.tailwindcss.com')) {
                  cache.put(event.request, responseToCache);
                }
              });

            return response;
          }
        ).catch(() => {
          // Offline fallback for API requests
          if (event.request.url.includes('/predict')) {
            return new Response(
              JSON.stringify({
                error: 'Offline - Prediction requires internet connection',
                offline: true
              }),
              {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
              }
            );
          }
          
          // For other requests, try to serve offline page
          if (event.request.mode === 'navigate') {
            return caches.match('/');
          }
        });
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
