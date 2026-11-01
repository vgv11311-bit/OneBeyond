// Also bound the UI callback: callers other than api.ts must not leave it pending.
export const RUNTIME_CHECK_LIMIT_MS = 50000;

export async function checkRuntimeWithDeadline(check: () => Promise<void>): Promise<void> {
  let timer: ReturnType<typeof setTimeout>;
  try {
    await Promise.race([
      Promise.resolve().then(check),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('GPU check timed out. Loading has stopped; use Reconnect running GPU to retry.')), RUNTIME_CHECK_LIMIT_MS);
      }),
    ]);
  } finally { clearTimeout(timer!); }
}
