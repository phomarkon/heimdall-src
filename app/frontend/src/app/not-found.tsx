import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f4f7fb] px-6 text-slate-900">
      <section className="max-w-md rounded-md border border-[#d8e2ec] bg-white p-6 shadow-sm">
        <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Heimdall</p>
        <h1 className="mt-2 text-xl font-semibold">View not found</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          This MVP currently exposes the operator dashboard as the single primary view.
        </p>
        <Link
          href="/"
          className="mt-4 inline-flex rounded-md border border-teal-200 bg-teal-50 px-3 py-2 text-sm font-medium text-teal-700 transition hover:border-teal-300"
        >
          Return to dashboard
        </Link>
      </section>
    </main>
  );
}
