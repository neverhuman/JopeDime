import { describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import App from "./App";
import { LossChart } from "./App";
import { bytes, count, validateSnapshot } from "./types";
import type { Point } from "./types";
describe("Evidence semantics", () => {
  it("keeps unknown distinct from measured zero", () => {
    expect(count(null)).toBe("—");
    expect(count(0)).toBe("0");
    expect(bytes(undefined)).toBe("—");
    expect(bytes(0)).toBe("0 B");
    expect(bytes(1024)).toBe("1 KiB");
  });
  it("rejects a different or incomplete observer schema", () => {
    expect(() => validateSnapshot({ schema: "legacy/v1" })).toThrow();
    expect(() =>
      validateSnapshot({ schema: "jope.observatory/v1", runs: [] }),
    ).toThrow();
  });
  it("does not claim training when no qualified curve exists", () => {
    render(<LossChart points={[]} />);
    expect(screen.getByText(/No identity-qualified/)).toBeInTheDocument();
    expect(screen.queryByRole("img")).toBeNull();
  });
  it("breaks the plotted series when panel identity changes", () => {
    const point = (step: number, panel: string): Point => ({
      step,
      train: 4 - step,
      validation: 4.5 - step,
      at: 1,
      panel_sha256: panel,
      objective: "fixed",
      terms: { training: {}, validation: {} },
    });
    const { container } = render(
      <LossChart points={[point(0, "a"), point(1, "a"), point(2, "b")]} />,
    );
    expect(container.querySelectorAll(".train-line")).toHaveLength(2);
    expect(container.querySelectorAll(".validation-line")).toHaveLength(2);
    expect(
      screen.getByText(/Full validation and verified self-play/),
    ).toBeInTheDocument();
  });
  it("keeps the last real observation visible when live fetch fails", async () => {
    vi.useFakeTimers();
    const snapshot = {
      schema: "jope.observatory/v1",
      generated_at: Date.now() / 1000,
      service: {
        version: "0.3.0",
        mode: "read-only",
        refresh_seconds: 5,
        production_authority: false,
      },
      runs: [],
      storage: {
        catalog_fallback: false,
        evidence: {
          status: "observed",
          value: {
            buckets: [],
            observed_bytes: null,
            observed_objects: null,
            complete: false,
            scope: "partial",
          },
        },
      },
      datasets: [],
      fleet: [],
      gp: {
        unique_programs: 6899,
        sources: [],
        families: {},
        status: "observed",
      },
      research: { waves: [] },
      limits: {},
    };
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce({ ok: true, json: async () => snapshot })
        .mockRejectedValue(new Error("Network unavailable")),
    );
    const view = render(<App />);
    try {
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(screen.getByText("6,899")).toBeInTheDocument();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Retaining the last observation",
      );
      expect(screen.getByText("6,899")).toBeInTheDocument();
    } finally {
      view.unmount();
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });
});
