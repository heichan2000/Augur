export default function Home() {
  return (
    <div className="flex flex-1 items-center justify-center bg-zinc-50 dark:bg-black">
      <main className="flex flex-col items-center gap-4 text-center px-8">
        <h1 className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
          Augur
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          Developer-docs AI assistant — chat UI coming in Phase 1.
        </p>
      </main>
    </div>
  );
}
