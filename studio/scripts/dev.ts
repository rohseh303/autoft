// One command, two processes: the Elysia BFF (Bun, :8787) + the Vite client (:5173).
// Vite proxies /api -> :8787. Bun-native spawn so it works on any shell.
import { spawn } from "bun";

const procs = [
  spawn({ cmd: ["bun", "--watch", "server/index.ts"], stdout: "inherit", stderr: "inherit", stdin: "inherit" }),
  spawn({ cmd: ["bunx", "vite"], stdout: "inherit", stderr: "inherit", stdin: "inherit" }),
];

const shutdown = () => {
  for (const p of procs) p.kill();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

await Promise.race(procs.map((p) => p.exited));
shutdown();
