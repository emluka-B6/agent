import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { TaskNodeData } from "../api/graph";

export type TaskFlowNode = Node<TaskNodeData, "task">;

export default function TaskNode({ data, selected }: NodeProps<TaskFlowNode>) {
  return (
    <div
      className={`task-node complexity-${data.complexity}${
        selected ? " is-selected" : ""
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <span className="task-node-id">{data.id}</span>
      <span className="task-node-title">{data.title}</span>
      <span className="task-node-meta">{data.complexity}</span>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
