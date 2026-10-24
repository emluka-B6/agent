import { z } from "zod";

import {
  graphSchema,
  planDocumentSchema,
  planListSchema,
  type GraphPayload,
  type PlanDocument,
  type PlanList,
} from "./types";

export type RemoveStrategy = "rewire" | "block" | "cascade";

export class ApiError extends Error {
  readonly status: number;
  readonly blockers: string[];
  readonly unknownIds: string[];

  constructor(
    message: string,
    status: number,
    blockers: string[] = [],
    unknownIds: string[] = [],
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.blockers = blockers;
    this.unknownIds = unknownIds;
  }
}

interface ErrorBody {
  detail?: unknown;
  blockers?: string[];
  unknown_ids?: string[];
}

function describeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (item && typeof item === "object" && "msg" in item) {
        return String((item as { msg: unknown }).msg);
      }
      return String(item);
    });
    return messages.join("; ") || fallback;
  }
  return fallback;
}

async function parseError(response: Response): Promise<ApiError> {
  const body = (await response.json().catch(() => ({}))) as ErrorBody;
  return new ApiError(
    describeDetail(body.detail, response.statusText),
    response.status,
    body.blockers ?? [],
    body.unknown_ids ?? [],
  );
}

interface Parser<T> {
  parse(data: unknown): T;
}

async function requestJson<T>(
  url: string,
  schema: Parser<T>,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...init, headers });
  if (!response.ok) {
    throw await parseError(response);
  }
  return schema.parse(await response.json());
}

export function listPlans(): Promise<PlanList> {
  return requestJson("/api/plans", planListSchema);
}

export function loadPlan(name: string): Promise<PlanDocument> {
  return requestJson(
    `/api/plans/${encodeURIComponent(name)}`,
    planDocumentSchema,
  );
}

/** Graph endpoint, kept for API consumers that do not derive the DAG locally. */
export function loadGraph(name: string): Promise<GraphPayload> {
  return requestJson(
    `/api/plans/${encodeURIComponent(name)}/graph`,
    graphSchema,
  );
}

export function removeSubtasks(
  name: string,
  document: PlanDocument,
  ids: string[],
  strategy: RemoveStrategy,
): Promise<PlanDocument> {
  return requestJson(
    `/api/plans/${encodeURIComponent(name)}/remove`,
    planDocumentSchema,
    {
      method: "POST",
      body: JSON.stringify({ ids, strategy, document }),
    },
  );
}

const saveResultSchema = z.object({ path: z.string(), format: z.string() });
export type SaveResult = z.infer<typeof saveResultSchema>;

export function savePlan(
  name: string,
  document: PlanDocument,
): Promise<SaveResult> {
  return requestJson(
    `/api/plans/${encodeURIComponent(name)}/save`,
    saveResultSchema,
    {
      method: "POST",
      body: JSON.stringify({ document }),
    },
  );
}

export interface ExportResult {
  blob: Blob;
  filename: string;
}

export async function exportPlan(
  name: string,
  document: PlanDocument,
  format: "json" | "markdown",
): Promise<ExportResult> {
  const response = await fetch(
    `/api/plans/${encodeURIComponent(name)}/export`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document, format }),
    },
  );
  if (!response.ok) {
    throw await parseError(response);
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = /filename="([^"]+)"/.exec(disposition);
  return {
    blob: await response.blob(),
    filename:
      match?.[1] ?? (format === "markdown" ? "plan.md" : "plan.json"),
  };
}
