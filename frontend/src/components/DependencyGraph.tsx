import {
  useEffect,
  useMemo,
  useRef,
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  type Edge,
  type EdgeMouseHandler,
  type NodeMouseHandler,
  type XYPosition,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";

import {
  documentEdges,
  documentNodes,
  simplifyEdges,
} from "../api/graph";
import type { PlanDocument } from "../api/types";
import MaximizeButton from "./MaximizeButton";
import TaskNode, { type TaskFlowNode } from "./TaskNode";

// The default graph font size corresponds to the original 236x88 node box.
// Bigger labels scale the box with it so titles keep the same line count.
const BASE_FONT_SIZE = 12;
const NODE_WIDTH_PER_FONT_SIZE = 236 / BASE_FONT_SIZE;
const NODE_HEIGHT_PER_FONT_SIZE = 88 / BASE_FONT_SIZE;
const FIT_VIEW_OPTIONS = { padding: 0.2 };
const nodeTypes = { task: TaskNode };

// The arrowhead is drawn inside a -10..10 viewBox, so the visible triangle is
// about a quarter of the marker width and 40% of its height. React Flow's
// default 12.5px marker therefore renders a ~5px arrow that is easy to miss.
// The larger marker keeps the dependency direction readable; ``userSpaceOnUse``
// makes its size independent from the edge stroke width, so selecting an edge
// (which thickens the line) does not blow the arrowhead up.
const EDGE_MARKER = {
  type: MarkerType.ArrowClosed,
  width: 28,
  height: 28,
  markerUnits: "userSpaceOnUse",
};
// Matches ``--accent`` in styles.css, which is also the border color of a
// selected task node. SVG markers need a literal color because their id is
// derived from the edge props.
const SELECTED_EDGE_MARKER_COLOR = "#2563eb";

interface Props {
  document: PlanDocument;
  selectedId: string | null;
  fontSize: number;
  maximized: boolean;
  onSelect: (id: string) => void;
  onRequestRemove: (ids: string[]) => void;
  onToggleMaximize: () => void;
}

function nodeSize(fontSize: number): { width: number; height: number } {
  return {
    width: Math.round(fontSize * NODE_WIDTH_PER_FONT_SIZE),
    height: Math.round(fontSize * NODE_HEIGHT_PER_FONT_SIZE),
  };
}

type PortSide = "source" | "target";

// Clicking this close (in screen pixels) to the end of a line selects every
// line attached to that task port. Further away, the graph keeps the classic
// single-line selection so one dependency can still be inspected on its own.

const PORT_SELECTION_RADIUS = 28;

/**
 * Which task port of the clicked line the pointer is closest to.
 *
 * Tasks are laid out top to bottom, so a line starts at the output (source)
 * handle of the task above it and ends at the input (target) handle of the
 * task below. React Flow reports the click in screen coordinates, while the
 * points of the SVG path use layout coordinates, so the path ends go through
 * the same viewport transform React Flow uses for the canvas. ``null`` means
 * the click was too far from either port to belong to a port group.
 */
function nearestPort(
  event: ReactMouseEvent,
  path: SVGPathElement,
  flowToScreenPosition: (position: XYPosition) => XYPosition,
): PortSide | null {
  const start = path.getPointAtLength(0);
  const end = path.getPointAtLength(path.getTotalLength());
  const startScreen = flowToScreenPosition({ x: start.x, y: start.y });
  const endScreen = flowToScreenPosition({ x: end.x, y: end.y });
  const toStart = Math.hypot(
    event.clientX - startScreen.x,
    event.clientY - startScreen.y,
  );
  const toEnd = Math.hypot(
    event.clientX - endScreen.x,
    event.clientY - endScreen.y,
  );
  //Remove to allow selection of all port edges even when the click is far from the port.
  if (Math.min(toStart, toEnd) > PORT_SELECTION_RADIUS) {
    return null;
  }
  return toStart <= toEnd ? "source" : "target";
}

/**
 * The lines that share the clicked line's port on one side.
 *
 * Grouping by ``source`` (plus its handle) collects every task that depends on
 * the same task, grouping by ``target`` collects every dependency of one task.
 * That is the one-to-many relation a single React Flow edge selection cannot
 * show.
 */
function portEdgeIds(
  edges: Edge[],
  edge: Edge,
  side: PortSide,
): Set<string> {
  const nodeId = side === "source" ? edge.source : edge.target;
  const handle =
    (side === "source" ? edge.sourceHandle : edge.targetHandle) ?? null;
  return new Set(
    edges
      .filter((candidate) =>
        side === "source"
          ? candidate.source === nodeId &&
            (candidate.sourceHandle ?? null) === handle
          : candidate.target === nodeId &&
            (candidate.targetHandle ?? null) === handle,
      )
      .map((candidate) => candidate.id),
  );
}

function buildGraph(
  document: PlanDocument,
  fontSize: number,
): { nodes: TaskFlowNode[]; edges: Edge[]; hiddenEdges: number } {
  const layout = new dagre.graphlib.Graph();
  layout.setDefaultEdgeLabel(() => ({}));
  layout.setGraph({
    rankdir: "TB",
    nodesep: 48,
    ranksep: 76,
    marginx: 24,
    marginy: 24,
  });

  const { width: nodeWidth, height: nodeHeight } = nodeSize(fontSize);
  const taskNodes = documentNodes(document);
  const rawEdges = documentEdges(document);
  // The graph always shows the transitive reduction: implied links would make
  // the actual dependency direction harder to follow. Simplification is
  // presentation only: the plan document keeps every ``depends_on`` entry and
  // therefore saves/exports unchanged.
  const taskEdges = simplifyEdges(rawEdges);

  for (const node of taskNodes) {
    layout.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  }
  for (const edge of taskEdges) {
    layout.setEdge(edge.source, edge.target);
  }
  dagre.layout(layout);

  const nodes = taskNodes.map<TaskFlowNode>((node) => {
    const position = layout.node(node.id) as { x: number; y: number };
    return {
      id: node.id,
      type: "task",
      position: {
        x: position.x - nodeWidth / 2,
        y: position.y - nodeHeight / 2,
      },
      data: node,
    };
  });

  const edges = taskEdges.map<Edge>((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: "smoothstep",
    markerEnd: EDGE_MARKER,
    // Selection is handled in GraphView: clicking a line highlights every line
    // that shares one of its task ports. Turning off the built-in per-edge
    // selection keeps React Flow from replacing that group with the one line
    // the pointer happened to hit. A slightly wider invisible stroke makes the
    // lines easier to hit near a port.
    selectable: false,
    interactionWidth: 24,
  }));

  return { nodes, edges, hiddenEdges: rawEdges.length - taskEdges.length };
}

function GraphView({
  document,
  selectedId,
  fontSize,
  maximized,
  onSelect,
  onRequestRemove,
  onToggleMaximize,
}: Props) {
  const {
    nodes: layoutNodes,
    edges: layoutEdges,
    hiddenEdges,
  } = useMemo(() => buildGraph(document, fontSize), [document, fontSize]);
  const canvasStyle = useMemo<CSSProperties>(() => {
    const { width, height } = nodeSize(fontSize);
    return {
      "--node-width": `${width}px`,
      "--node-height": `${height}px`,
    } as CSSProperties;
  }, [fontSize]);
  const [nodes, setNodes, onNodesChange] =
    useNodesState<TaskFlowNode>(layoutNodes);
  // Edges are controlled as well: the selected group is restyled below and
  // clicking the pane or a task clears it through ``onEdgesChange``.
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(layoutEdges);
  const { fitView, flowToScreenPosition } = useReactFlow<TaskFlowNode, Edge>();
  const nodesInitialized = useNodesInitialized();

  // Rebuild the nodes whenever the plan changes or a subtask is removed.
  useEffect(() => {
    setNodes(layoutNodes);
  }, [layoutNodes, setNodes]);

  // The same applies to the edges: a new layout replaces the old connections,
  // and the edge selection kept by React Flow restarts with it.
  useEffect(() => {
    setEdges(layoutEdges);
  }, [layoutEdges, setEdges]);

  // Arrowheads are shared <marker> definitions, so the selected edge needs its
  // own marker to be drawn in the same accent as its path and the border of the
  // selected task node.
  const displayEdges = useMemo(
    () =>
      edges.map((edge) =>
        edge.selected
          ? {
              ...edge,
              markerEnd: { ...EDGE_MARKER, color: SELECTED_EDGE_MARKER_COLOR },
            }
          : edge,
      ),
    [edges],
  );

  // React Flow keeps selection in its own store. Mirror the selection owned by
  // the parent so picking a task in the list (or having it re-mapped after a
  // removal) highlights the same node in the graph.
  useEffect(() => {
    setNodes((current) => {
      let changed = false;
      const next = current.map((node) => {
        const selected = node.id === selectedId;
        if ((node.selected ?? false) === selected) {
          return node;
        }
        changed = true;
        return { ...node, selected };
      });
      return changed ? next : current;
    });
    // layoutNodes is listed so the selection is re-applied to freshly built
    // nodes after the effect above replaces them.
  }, [layoutNodes, selectedId, setNodes]);

  // Positions are derived from the document, so only refit the viewport when
  // the layout actually changes. Waiting for initialization makes sure the
  // fit uses the measured node sizes.
  const graphKey = useMemo(
    () =>
      layoutNodes
        .map((node) => `${node.id}:${node.position.x},${node.position.y}`)
        .join("|"),
    [layoutNodes],
  );
  const fittedKey = useRef<string | null>(null);

  useEffect(() => {
    if (!nodesInitialized || fittedKey.current === graphKey) {
      return;
    }
    fittedKey.current = graphKey;
    void fitView(FIT_VIEW_OPTIONS);
  }, [fitView, graphKey, nodesInitialized]);

  // Maximizing changes the canvas size, which React Flow measures
  // asynchronously. Refit after two frames so the graph uses the new viewport.
  useEffect(() => {
    if (!nodesInitialized) {
      return;
    }
    let secondFrame = 0;
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => {
        void fitView(FIT_VIEW_OPTIONS);
      });
    });
    return () => {
      cancelAnimationFrame(firstFrame);
      cancelAnimationFrame(secondFrame);
    };
  }, [fitView, maximized, nodesInitialized]);

  const handleNodeClick: NodeMouseHandler = (_event, node) => {
    onSelect(node.id);
  };

  // A task often has several dependencies and several dependents, so clicking
  // a line close to one of its task ports highlights every line attached to
  // that port instead of just the line that was hit. A click further from a
  // port selects the single line, and clicking one of the highlighted lines
  // again removes only that line from the selection.
  const handleEdgeClick: EdgeMouseHandler = (event, edge) => {
    const element = event.currentTarget as SVGGElement;
    const path = element.querySelector<SVGPathElement>(
      ".react-flow__edge-path",
    );
    const side = path
      ? nearestPort(event, path, flowToScreenPosition)
      : null;
    setEdges((current) => {
      const clicked = current.find((candidate) => candidate.id === edge.id);
      if (!clicked) {
        return current;
      }
      if (clicked.selected) {
        return current.map((candidate) =>
          candidate.id === edge.id
            ? { ...candidate, selected: false }
            : candidate,
        );
      }
      const group = side
        ? portEdgeIds(current, clicked, side)
        : new Set([clicked.id]);
      let changed = false;
      const next = current.map((candidate) => {
        const selected = group.has(candidate.id);
        if ((candidate.selected ?? false) === selected) {
          return candidate;
        }
        changed = true;
        return { ...candidate, selected };
      });
      return changed ? next : current;
    });
  };

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if ((event.key === "Delete" || event.key === "Backspace") && selectedId) {
      event.preventDefault();
      onRequestRemove([selectedId]);
    }
  }

  return (
    <section
      className={`panel graph-panel${maximized ? " maximized" : ""}`}
      aria-label="Dependency graph"
    >
      <header className="panel-header">
        <h2>Dependencies</h2>
        <div className="panel-actions">
          <span className="hint">
            {hiddenEdges > 0
              ? `${hiddenEdges} implied link${hiddenEdges === 1 ? "" : "s"} hidden · `
              : ""}
            Click a line near a port to select all links of that port · Click a
            task, then press Delete to remove it
          </span>
          <MaximizeButton
            expanded={maximized}
            label="dependency graph"
            onToggle={onToggleMaximize}
          />
        </div>
      </header>
      <div
        className="graph-canvas"
        style={canvasStyle}
        tabIndex={0}
        onKeyDown={handleKeyDown}
      >
        <ReactFlow
          nodes={nodes}
          edges={displayEdges}
          nodeTypes={nodeTypes}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          edgesFocusable={false}
          deleteKeyCode={null}
          minZoom={0.3}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </section>
  );
}

export default function DependencyGraph(props: Props) {
  return (
    <ReactFlowProvider>
      <GraphView {...props} />
    </ReactFlowProvider>
  );
}
