import type { ReactNode } from "react";

import type { PlanDocument } from "../api/types";

function Bullets({
  items,
  empty = "none",
}: {
  items: string[];
  empty?: string;
}) {
  if (items.length === 0) {
    return <p className="muted">{empty}</p>;
  }
  return (
    <ul className="bullets">
      {items.map((item, index) => (
        <li key={`${index}-${item}`}>{item}</li>
      ))}
    </ul>
  );
}

function Section({
  title,
  children,
  open = false,
}: {
  title: string;
  children: ReactNode;
  open?: boolean;
}) {
  return (
    <details className="section" open={open}>
      <summary>{title}</summary>
      <div className="section-body">{children}</div>
    </details>
  );
}

export default function SectionsPanel({
  document,
}: {
  document: PlanDocument;
}) {
  const review = document.review;
  const research = document.research;
  const metadata = document.metadata as Record<string, unknown>;
  const edits = Array.isArray(metadata.edits)
    ? (metadata.edits as Array<Record<string, unknown>>)
    : [];

  return (
    <section className="sections" aria-label="Plan context">
      <Section title="Summary" open>
        <p>{document.summary || "—"}</p>
        <p className="muted">Issue: {document.issue || "—"}</p>
      </Section>

      <Section title={`Assumptions (${document.assumptions.length})`}>
        <Bullets items={document.assumptions} />
      </Section>

      <Section title={`Open questions (${document.open_questions.length})`}>
        <Bullets items={document.open_questions} />
      </Section>

      <Section title={`Definition of done (${document.definition_of_done.length})`}>
        <Bullets items={document.definition_of_done} />
      </Section>

      {review && (
        <Section
          title={`Review — ${review.approved ? "approved" : "not approved"}`}
        >
          <p>{review.summary || "—"}</p>
          <h4>Blocking issues</h4>
          <Bullets items={review.blocking_issues} />
          <h4>Suggestions</h4>
          <Bullets items={review.suggestions} />
        </Section>
      )}

      {research && (
        <Section title="Research">
          {research.error && (
            <p className="error-text">Limitation: {research.error}</p>
          )}
          {research.summary ? (
            <>
              <h4>Key facts</h4>
              <Bullets items={research.summary.key_facts} />
              <h4>Constraints</h4>
              <Bullets items={research.summary.constraints} />
              <h4>Recommended approach</h4>
              <p>{research.summary.recommended_approach || "—"}</p>
              <h4>Research assumptions</h4>
              <Bullets items={research.summary.assumptions} />
            </>
          ) : (
            !research.error && <p className="muted">No research summary.</p>
          )}
          <h4>Sources ({research.sources.length})</h4>
          {research.sources.length === 0 ? (
            <p className="muted">none</p>
          ) : (
            <ul className="bullets">
              {research.sources.map((source, index) => (
                <li key={`${index}-${source.url}`}>
                  {source.url ? (
                    <a href={source.url} target="_blank" rel="noreferrer">
                      {source.title || source.url}
                    </a>
                  ) : (
                    source.title || "source"
                  )}
                  {source.snippet && (
                    <span className="muted"> — {source.snippet}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Section>
      )}

      <Section title="Metadata">
        <pre className="code">{JSON.stringify(metadata, null, 2)}</pre>
      </Section>

      {edits.length > 0 && (
        <Section title={`Edit history (${edits.length})`}>
          <ul className="bullets">
            {edits.map((edit, index) => (
              <li key={`${index}-${String(edit.at)}`}>
                <code>{String(edit.at ?? "unknown time")}</code> —{" "}
                {String(edit.operation ?? "edit")}
                {Array.isArray(edit.ids) ? ` ${edit.ids.join(", ")}` : ""}
                {edit.strategy ? ` (${String(edit.strategy)})` : ""}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </section>
  );
}
