import type { Complexity, PlanDocument } from "./types";

export interface TaskNodeData {
  // React Flow requires node data to be a string-keyed record.
  [key: string]: unknown;
  id: string;
  title: string;
  complexity: Complexity;
}

export interface TaskEdgeData {
  id: string;
  source: string;
  target: string;
}

export function documentNodes(document: PlanDocument): TaskNodeData[] {
  return document.subtasks.map((task) => ({
    id: task.id,
    title: task.title,
    complexity: task.complexity,
  }));
}

export function documentEdges(document: PlanDocument): TaskEdgeData[] {
  const known = new Set(document.subtasks.map((task) => task.id));
  const edges: TaskEdgeData[] = [];
  for (const task of document.subtasks) {
    // Duplicate ids in ``depends_on`` would render duplicate React Flow edges.
    for (const dependency of new Set(task.depends_on)) {
      if (known.has(dependency)) {
        edges.push({
          id: `${dependency}->${task.id}`,
          source: dependency,
          target: task.id,
        });
      }
    }
  }
  return edges;
}

/**
 * Transitive reduction of the dependency edges, for presentation only.
 *
 * An edge ``A -> B`` is redundant when ``B`` is also reachable from ``A``
 * through another task: if ``T9`` depends on ``T8``, ``T8`` depends on
 * ``T2``, and ``T9`` also lists ``T2`` directly, only ``T8 -> T9`` is kept.
 * The document itself (and therefore ``depends_on``) is never modified.
 *
 * The traversal never revisits the edge's source, so degenerate cyclic input
 * cannot produce an infinite loop. When every dependency of a task looks
 * redundant (possible only with cyclic data), all of its edges are kept.
 */
export function simplifyEdges(edges: TaskEdgeData[]): TaskEdgeData[] {
  const adjacency = new Map<string, Set<string>>();
  for (const edge of edges) {
    const targets = adjacency.get(edge.source);
    if (targets) {
      targets.add(edge.target);
    } else {
      adjacency.set(edge.source, new Set([edge.target]));
    }
  }

  function hasIndirectPath(source: string, target: string): boolean {
    const visited = new Set<string>([source]);
    const stack = [...(adjacency.get(source) ?? [])].filter(
      (next) => next !== target,
    );
    while (stack.length > 0) {
      const node = stack.pop() as string;
      if (node === target) {
        return true;
      }
      if (visited.has(node)) {
        continue;
      }
      visited.add(node);
      for (const next of adjacency.get(node) ?? []) {
        if (!visited.has(next)) {
          stack.push(next);
        }
      }
    }
    return false;
  }

  const kept = edges.filter((edge) => !hasIndirectPath(edge.source, edge.target));

  // Keep every edge for a task whose whole dependency list vanished; that can
  // only happen with mutually cyclic dependencies.
  const keptIds = new Set(kept.map((edge) => edge.id));
  const keptByTarget = new Map<string, number>();
  for (const edge of kept) {
    keptByTarget.set(edge.target, (keptByTarget.get(edge.target) ?? 0) + 1);
  }
  for (const edge of edges) {
    if (
      keptIds.has(edge.id) ||
      (keptByTarget.get(edge.target) ?? 0) > 0
    ) {
      continue;
    }
    kept.push(edge);
    keptIds.add(edge.id);
    keptByTarget.set(edge.target, 1);
  }
  return kept;
}

/** Remaining subtasks that depend on any of the removed ids. */
export function findDependents(
  document: PlanDocument,
  removedIds: Iterable<string>,
): Record<string, string[]> {
  const removed = new Set(removedIds);
  const dependents: Record<string, string[]> = {};
  for (const task of document.subtasks) {
    if (removed.has(task.id)) {
      continue;
    }
    const blocking = task.depends_on.filter((dependency) =>
      removed.has(dependency),
    );
    if (blocking.length > 0) {
      dependents[task.id] = blocking;
    }
  }
  return dependents;
}

/** Every id that a cascade removal would delete, including the selection. */
export function cascadeIds(
  document: PlanDocument,
  selectedIds: Iterable<string>,
): string[] {
  const removed = new Set(selectedIds);
  let changed = true;
  while (changed) {
    changed = false;
    for (const task of document.subtasks) {
      if (removed.has(task.id)) {
        continue;
      }
      if (task.depends_on.some((dependency) => removed.has(dependency))) {
        removed.add(task.id);
        changed = true;
      }
    }
  }
  return [...removed];
}
