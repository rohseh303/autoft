import { useEffect, useRef } from "react";
import type { LogLine } from "../lib/useRunStream";

// The raw container log, streamed live from `modal app logs` (real) or a
// synthesized trace (mock). Auto-scrolls to the tail unless the user has
// scrolled up to read history.
export function LogStream({ lines, done }: { lines: LogLine[]; done: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Track whether the user is scrolled to the bottom; only auto-scroll if so.
  function onScroll() {
    const el = ref.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }

  useEffect(() => {
    const el = ref.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div className="ls" ref={ref} onScroll={onScroll}>
      {lines.length === 0 && <div className="ls-empty mono">waiting for container logs…</div>}
      {lines.map((l, i) => (
        <div
          key={i}
          className={`ls-line mono${l.stream === "stderr" ? " ls-err" : ""}${l.stream === "meta" ? " ls-meta" : ""}`}
        >
          {l.line}
        </div>
      ))}
      {done && <div className="ls-line mono ls-meta">— end of logs —</div>}
    </div>
  );
}
