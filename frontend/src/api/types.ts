import { z } from "zod";

export const complexitySchema = z.enum(["small", "medium", "large"]);
export type Complexity = z.infer<typeof complexitySchema>;

// ``passthrough`` keeps fields that a newer agent version might add, so that
// saving an edited document does not silently drop data.
export const subTaskSchema = z
  .object({
    id: z.string(),
    title: z.string(),
    description: z.string().default(""),
    implementation_prompt: z.string().default(""),
    depends_on: z.array(z.string()).default([]),
    files: z.array(z.string()).default([]),
    acceptance_criteria: z.array(z.string()).default([]),
    verification: z.string().default(""),
    complexity: complexitySchema.default("medium"),
    risks: z.array(z.string()).default([]),
  })
  .passthrough();
export type SubTask = z.infer<typeof subTaskSchema>;

export const sourceSchema = z
  .object({
    title: z.string().default(""),
    url: z.string().default(""),
    snippet: z.string().default(""),
  })
  .passthrough();
export type Source = z.infer<typeof sourceSchema>;

export const researchSummarySchema = z
  .object({
    key_facts: z.array(z.string()).default([]),
    constraints: z.array(z.string()).default([]),
    recommended_approach: z.string().default(""),
    assumptions: z.array(z.string()).default([]),
  })
  .passthrough();
export type ResearchSummary = z.infer<typeof researchSummarySchema>;

export const researchSchema = z
  .object({
    summary: researchSummarySchema.nullable().optional(),
    sources: z.array(sourceSchema).default([]),
    error: z.string().nullable().optional(),
  })
  .passthrough();
export type Research = z.infer<typeof researchSchema>;

export const reviewSchema = z
  .object({
    approved: z.boolean().default(false),
    summary: z.string().default(""),
    blocking_issues: z.array(z.string()).default([]),
    suggestions: z.array(z.string()).default([]),
  })
  .passthrough();
export type Review = z.infer<typeof reviewSchema>;

// Question objects carry one to four candidate answers; older artifacts stored
// a bare string, which is upgraded here so the viewer reads both.
export const openQuestionSchema = z.union([
  z.string().transform((question) => ({
    question,
    options: [] as string[],
    recommended: "",
    answer: null as string | null,
  })),
  z.object({
      question: z.string(),
      options: z.array(z.string()).default([]),
      recommended: z.string().default(""),
      answer: z.string().nullable().optional(),
    })
    .passthrough(),
]);
export type OpenQuestion = z.infer<typeof openQuestionSchema>;

// Older artifacts stored the brief under "issue". Both spellings are read and
// normalised to "brief", so the viewer opens old and new files while always
// writing the current shape back.
export const planDocumentSchema = z
  .object({
    brief: z.string().default(""),
    issue: z.string().optional(),
    generated_at: z.string().optional(),
    status: z.string().default("unknown"),
    approved: z.boolean().default(false),
    summary: z.string().default(""),
    subtasks: z.array(subTaskSchema),
    assumptions: z.array(z.string()).default([]),
    open_questions: z.array(openQuestionSchema).default([]),
    definition_of_done: z.array(z.string()).default([]),
    review: reviewSchema.nullable().optional(),
    research: researchSchema.optional(),
    metadata: z.record(z.unknown()).default({}),
  })
  .passthrough()
  .transform(({ issue, ...document }) => ({
    ...document,
    brief: document.brief || issue || "",
  }));
export type PlanDocument = z.infer<typeof planDocumentSchema>;

export const planSummarySchema = z.object({
  name: z.string(),
  // Human-readable identifier (the brief text) added by newer viewers.
  title: z.string().optional(),
  size: z.number(),
  modified_at: z.string(),
  generated_at: z.string().nullable().optional(),
  status: z.string().optional(),
  domain: z.string().optional(),
});
export type PlanSummary = z.infer<typeof planSummarySchema>;

export const planListSchema = z.object({ plans: z.array(planSummarySchema) });
export type PlanList = z.infer<typeof planListSchema>;

export const graphNodeSchema = z.object({
  id: z.string(),
  title: z.string(),
  complexity: z.string(),
});
export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
});
export const graphSchema = z.object({
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
});
export type GraphPayload = z.infer<typeof graphSchema>;
