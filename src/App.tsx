import { useCallback, useEffect, useRef, useState } from 'react';

const filmUrl = `${import.meta.env.BASE_URL}newton_the_hidden_order.mp4`;

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState<string | null>(null);

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      setPlaybackError(null);
      if (video.error) video.load();
      if (video.ended) {
        video.currentTime = 0;
      }
      video.play().catch((error: DOMException) => {
        if (error.name !== 'AbortError') {
          setPlaybackError('The film could not be played. Try again.');
        }
      });
    } else {
      video.pause();
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat && !(event.target instanceof HTMLButtonElement)) {
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
          src={filmUrl}
          playsInline
          preload="auto"
          className="w-full h-full object-contain bg-black block"
          onPlaying={() => {
            setIsPlaying(true);
            setPlaybackError(null);
          }}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          onError={() => {
            setIsPlaying(false);
            setPlaybackError('The film could not be loaded. Try again.');
          }}
        />

        {playbackError && (
          <p role="status" className="absolute inset-x-4 bottom-4 text-center text-sm text-white/80">
            {playbackError}{' '}
            <a href={filmUrl} download onClick={(event) => event.stopPropagation()} className="underline">
              Download MP4
            </a>
          </p>
        )}

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
