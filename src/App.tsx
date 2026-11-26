import { useCallback, useEffect, useRef, useState } from 'react';

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      if (video.ended) {
        video.currentTime = 0;
      }
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        event.preventDefault();
        togglePlayback();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [togglePlayback]);

  return (
    <main className="fixed inset-0 bg-black flex items-center justify-center overflow-hidden">
      <div
        className="relative w-full max-w-[177.7778vh] aspect-video bg-black flex items-center justify-center cursor-pointer"
        onClick={togglePlayback}
      >
        <video
          ref={videoRef}
          src="/newton_the_hidden_order.mp4"
          playsInline
          preload="auto"
          className="w-full h-full object-contain bg-black block"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
        />

        {!isPlaying && (
          <button
            type="button"
            aria-label="Play film"
            onClick={(e) => {
              e.stopPropagation();
              togglePlayback();
            }}
            className="absolute inset-0 m-auto w-20 h-20 rounded-full border border-white/30 bg-black/45 hover:bg-black/65 hover:border-white/55 transition-all duration-500 flex items-center justify-center focus:outline-none focus-visible:ring-1 focus-visible:ring-white/60 cursor-pointer"
          >
            <svg
              viewBox="0 0 24 24"
              className="w-7 h-7 fill-white/85 translate-x-0.5"
              aria-hidden="true"
            >
              <polygon points="6,4 20,12 6,20" />
            </svg>
          </button>
        )}
      </div>
    </main>
  );
}
