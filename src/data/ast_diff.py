"""
AST-Difference Graph construction (TG-CUP Section 2.4).
Uses javalang for Java AST parsing; builds multi-edge graph for GGNN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import javalang

# Edge types matching TG-CUP paper
EDGE_AST = 0       # ordinary AST parent-child
EDGE_COMPOUND = 1  # compound word split
EDGE_UPDATE = 2    # node updated old→new
EDGE_ORDER = 3     # sequential order in code


@dataclass
class ASTNode:
    node_id: int
    node_type: str          # AST structural type or "value"
    value: str              # token text (empty for structural nodes)
    is_value_node: bool     # light-gray nodes in paper – used as GGNN output


@dataclass
class ASTDiffGraph:
    nodes: List[ASTNode] = field(default_factory=list)
    edges: List[Tuple[int, int, int]] = field(default_factory=list)  # (src, dst, type)
    value_node_indices: List[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def adjacency(self, edge_type: int, num_edge_types: int = 4) -> List[List[int]]:
        """Return adjacency list per edge type for GGNN."""
        adj = [[] for _ in range(self.num_nodes)]
        for src, dst, et in self.edges:
            if et == edge_type:
                adj[src].append(dst)
        return adj

    def to_tensors(self, vocab_encode) -> Dict[str, Any]:
        import torch

        node_ids = []
        for n in self.nodes:
            text = n.value if n.is_value_node else n.node_type
            node_ids.append(vocab_encode(text))
        return {
            "node_ids": torch.tensor(node_ids, dtype=torch.long),
            "value_mask": torch.tensor(
                [1 if n.is_value_node else 0 for n in self.nodes], dtype=torch.bool
            ),
            "edges": self.edges,
            "num_nodes": self.num_nodes,
        }


def _parse_ast(code: str) -> Optional[javalang.tree.CompilationUnit]:
    try:
        return javalang.parse.parse(code)
    except Exception:
        return None


def _collect_nodes(
    node: Any,
    nodes: List[ASTNode],
    edges: List[Tuple[int, int, int]],
    parent_id: int = -1,
    order_counter: List[int] = None,
) -> int:
    """DFS collect AST nodes; return node_id of current node."""
    if order_counter is None:
        order_counter = [0]

    node_type = type(node).__name__
    value = ""
    is_value = False

    if hasattr(node, "name") and node.name:
        value = str(node.name)
        is_value = True
    elif hasattr(node, "value") and node.value is not None:
        value = str(node.value)
        is_value = True
    elif hasattr(node, "member") and node.member:
        value = str(node.member)
        is_value = True

    nid = len(nodes)
    nodes.append(ASTNode(nid, node_type, value, is_value))

    if parent_id >= 0:
        edges.append((parent_id, nid, EDGE_AST))

    prev_sibling = -1
    for _field_name, child in node:
        if isinstance(child, list):
            for item in child:
                if isinstance(item, javalang.tree.Node):
                    child_id = _collect_nodes(item, nodes, edges, nid, order_counter)
                    if prev_sibling >= 0:
                        edges.append((prev_sibling, child_id, EDGE_ORDER))
                    prev_sibling = child_id
        elif isinstance(child, javalang.tree.Node):
            child_id = _collect_nodes(child, nodes, edges, nid, order_counter)
            if prev_sibling >= 0:
                edges.append((prev_sibling, child_id, EDGE_ORDER))
            prev_sibling = child_id

    return nid


def _node_signature(n: ASTNode) -> str:
    return f"{n.node_type}:{n.value}" if n.is_value_node else n.node_type


def build_ast_diff_graph(old_code: str, new_code: str) -> Optional[ASTDiffGraph]:
    """
    Build AST-Difference Graph from old/new Java method code.
    Returns None if parsing fails (sample should be filtered).
    """
    old_ast = _parse_ast(old_code)
    new_ast = _parse_ast(new_code)
    if old_ast is None or new_ast is None:
        return None

    old_nodes: List[ASTNode] = []
    old_edges: List[Tuple[int, int, int]] = []
    new_nodes: List[ASTNode] = []
    new_edges: List[Tuple[int, int, int]] = []

    _collect_nodes(old_ast, old_nodes, old_edges)
    _collect_nodes(new_ast, new_nodes, new_edges)

    # Align nodes by type+value signature
    old_sigs = [_node_signature(n) for n in old_nodes]
    new_sigs = [_node_signature(n) for n in new_nodes]

    diff_nodes: List[ASTNode] = []
    diff_edges: List[Tuple[int, int, int]] = []
    value_indices: List[int] = []

    old_matched = set()
    new_matched = set()

    # Keep matching nodes
    sig_to_old = {}
    for i, s in enumerate(old_sigs):
        sig_to_old.setdefault(s, []).append(i)

    old_to_diff: Dict[int, int] = {}
    new_to_diff: Dict[int, int] = {}

    for j, sig in enumerate(new_sigs):
        if sig in sig_to_old and sig_to_old[sig]:
            oi = sig_to_old[sig].pop(0)
            old_matched.add(oi)
            new_matched.add(j)
            nid = len(diff_nodes)
            diff_nodes.append(ASTNode(nid, new_nodes[j].node_type, new_nodes[j].value, new_nodes[j].is_value_node))
            old_to_diff[oi] = nid
            new_to_diff[j] = nid
            if new_nodes[j].is_value_node:
                value_indices.append(nid)

    # Deleted nodes (in old, not in new)
    for i, n in enumerate(old_nodes):
        if i not in old_matched:
            pass  # deleted – omitted from diff graph per TG-CUP

    # Inserted nodes (in new, not matched)
    for j, n in enumerate(new_nodes):
        if j not in new_matched:
            nid = len(diff_nodes)
            diff_nodes.append(ASTNode(nid, n.node_type, n.value, n.is_value_node))
            new_to_diff[j] = nid
            if n.is_value_node:
                value_indices.append(nid)

    # Updated nodes: same position/type but different value
    for oi, on in enumerate(old_nodes):
        if oi in old_matched:
            continue
        for nj, nn in enumerate(new_nodes):
            if nj in new_matched:
                continue
            if on.node_type == nn.node_type and on.value != nn.value:
                old_nid = len(diff_nodes)
                diff_nodes.append(ASTNode(old_nid, on.node_type, on.value, on.is_value_node))
                new_nid = len(diff_nodes)
                diff_nodes.append(ASTNode(new_nid, nn.node_type, nn.value, nn.is_value_node))
                diff_edges.append((old_nid, new_nid, EDGE_UPDATE))
                if on.is_value_node:
                    value_indices.append(old_nid)
                if nn.is_value_node:
                    value_indices.append(new_nid)
                old_matched.add(oi)
                new_matched.add(nj)
                break

    # Rebuild structural edges from new AST onto diff node ids
    for src, dst, et in new_edges:
        if src in new_to_diff and dst in new_to_diff:
            diff_edges.append((new_to_diff[src], new_to_diff[dst], et))

    if not diff_nodes:
        # fallback: single dummy value node
        diff_nodes.append(ASTNode(0, "Method", "method", True))
        value_indices = [0]

    return ASTDiffGraph(
        nodes=diff_nodes,
        edges=diff_edges,
        value_node_indices=value_indices,
    )
