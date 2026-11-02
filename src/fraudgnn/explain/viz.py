"""Interactive causal neighborhood visualization for flagged transactions.

Renders the 2-hop causal historical sub-network into standalone interactive HTML
(PyVis) or a static PNG (matplotlib). Colors target transactions, shared devices/IPs,
and user accounts.
Labels and fraud flags are never used as model inputs and appear strictly as an
optional, explicitly-labeled post-hoc overlay.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from pyvis.network import Network

# "remote" loads vis-network from a CDN (small pages, needs internet to view);
# "in_line" embeds it (offline, ~700 KB per page). Neither writes a local lib/ folder.
VIZ_RESOURCES = ("remote", "in_line")

NODE_COLORS = {
    "user": "#38BDF8",  # Light blue
    "merchant": "#A855F7",  # Purple
    "device": "#10B981",  # Emerald Green
    "ip": "#FB923C",  # Warm Orange
}
DEFAULT_NODE_COLOR = "#94A3B8"
TARGET_COLOR = "#E11D48"


def extract_2hop_causal_subgraph(
    df: pd.DataFrame,
    target_tx_id: str,
    max_neighbors_per_entity: int = 10,
) -> nx.MultiDiGraph:
    """Extract strictly causal 2-hop neighborhood graph for a target transaction.

    Only transactions strictly before the target's timestamp are included, plus the
    target transaction itself.

    Args:
        df: Transaction DataFrame containing timestamp, user_id, merchant_id, device_id, etc.
        target_tx_id: Identifier of the target transaction.
        max_neighbors_per_entity: Limit on historical neighbor links per entity.

    Returns:
        NetworkX MultiDiGraph representing the causal 2-hop topology.
    """
    target_row = df[df["transaction_id"] == target_tx_id]
    if target_row.empty:
        raise ValueError(f"Transaction ID '{target_tx_id}' not found in dataset.")

    row = target_row.iloc[0]
    t_target = pd.to_datetime(row["timestamp"])
    target_user = str(row["user_id"])
    target_merch = str(row["merchant_id"])
    target_dev = str(row["device_id"]) if pd.notna(row.get("device_id")) else None
    target_ip = str(row["ip_address"]) if pd.notna(row.get("ip_address")) else None

    # Filter strictly causal history up to t_target
    df_history = df[
        (pd.to_datetime(df["timestamp"]) < t_target) | (df["transaction_id"] == target_tx_id)
    ].copy()

    g = nx.MultiDiGraph()

    # 1. Add core target nodes
    g.add_node(
        f"user_{target_user}",
        label=f"User\n{target_user[:10]}",
        title=f"Target User: {target_user}",
        entity_type="user",
        is_target=True,
    )
    g.add_node(
        f"merch_{target_merch}",
        label=f"Merchant\n{target_merch[:10]}",
        title=f"Target Merchant: {target_merch}",
        entity_type="merchant",
        is_target=True,
    )

    # Core target transaction edge
    g.add_edge(
        f"user_{target_user}",
        f"merch_{target_merch}",
        key=f"tx_{target_tx_id}",
        relation="target_transacts_at",
        amount=float(row["amount"]),
        timestamp=str(row["timestamp"]),
        is_target_edge=True,
    )

    if target_dev:
        g.add_node(
            f"dev_{target_dev}",
            label=f"Device\n{target_dev[:10]}",
            title=f"Target Device: {target_dev}",
            entity_type="device",
            is_target=True,
        )
        g.add_edge(
            f"user_{target_user}",
            f"dev_{target_dev}",
            key=f"uses_dev_{target_tx_id}",
            relation="uses_device",
            is_target_edge=True,
        )

    if target_ip:
        g.add_node(
            f"ip_{target_ip}",
            label=f"IP\n{target_ip[:10]}",
            title=f"Target IP: {target_ip}",
            entity_type="ip",
            is_target=True,
        )
        g.add_edge(
            f"user_{target_user}",
            f"ip_{target_ip}",
            key=f"uses_ip_{target_tx_id}",
            relation="uses_ip",
            is_target_edge=True,
        )

    # 2. Hop 1 & 2: Find other users sharing this device or IP historically
    if target_dev:
        dev_shared = df_history[df_history["device_id"] == target_dev]
        other_dev_users = dev_shared["user_id"].unique()
        for u in other_dev_users[:max_neighbors_per_entity]:
            u_str = str(u)
            if u_str != target_user:
                g.add_node(
                    f"user_{u_str}",
                    label=f"User\n{u_str[:10]}",
                    title=f"Shared Device User: {u_str}",
                    entity_type="user",
                    is_target=False,
                )
                g.add_edge(
                    f"user_{u_str}",
                    f"dev_{target_dev}",
                    relation="uses_device",
                    is_target_edge=False,
                )

    if target_ip:
        ip_shared = df_history[df_history["ip_address"] == target_ip]
        other_ip_users = ip_shared["user_id"].unique()
        for u in other_ip_users[:max_neighbors_per_entity]:
            u_str = str(u)
            if u_str != target_user:
                g.add_node(
                    f"user_{u_str}",
                    label=f"User\n{u_str[:10]}",
                    title=f"Shared IP User: {u_str}",
                    entity_type="user",
                    is_target=False,
                )
                g.add_edge(
                    f"user_{u_str}",
                    f"ip_{target_ip}",
                    relation="uses_ip",
                    is_target_edge=False,
                )

    # 3. Add other merchants transacted by target user prior to t_target
    prior_txs = df_history[
        (df_history["user_id"] == target_user) & (df_history["transaction_id"] != target_tx_id)
    ]
    for _, p_row in prior_txs.tail(max_neighbors_per_entity).iterrows():
        m_str = str(p_row["merchant_id"])
        g.add_node(
            f"merch_{m_str}",
            label=f"Merchant\n{m_str[:10]}",
            title=f"Prior Merchant: {m_str}",
            entity_type="merchant",
            is_target=False,
        )
        g.add_edge(
            f"user_{target_user}",
            f"merch_{m_str}",
            relation="prior_transacts_at",
            amount=float(p_row["amount"]),
            timestamp=str(p_row["timestamp"]),
            is_target_edge=False,
        )

    return g


def render_causal_neighborhood_html(
    df: pd.DataFrame,
    target_tx_id: str,
    output_path: str,
    reason_codes: list[str] | None = None,
    risk_score: float | None = None,
    threshold: float = 0.50,
    post_hoc_fraud_type: str | None = None,
    is_fraud_ground_truth: bool | None = None,
    model_name: str = "Model",
    cdn_resources: str = "remote",
) -> str:
    """Render interactive PyVis HTML network of the 2-hop causal neighborhood.

    Args:
        df: Transaction dataset.
        target_tx_id: Target transaction ID.
        output_path: Destination HTML file path.
        reason_codes: Optional list of 3-5 explanation sentences.
        risk_score: Model predicted probability.
        threshold: Operating decision threshold.
        post_hoc_fraud_type: Ground truth simulation label for post-hoc display only.
        is_fraud_ground_truth: Ground truth fraud flag for post-hoc display only.
        model_name: Name of the single model whose score and reason codes are shown.
        cdn_resources: "remote" (CDN scripts, small file) or "in_line" (embedded scripts).

    Returns:
        HTML string content of the generated visualization.
    """
    if cdn_resources not in VIZ_RESOURCES:
        raise ValueError(f"cdn_resources must be one of {VIZ_RESOURCES}, got {cdn_resources!r}")
    g = extract_2hop_causal_subgraph(df, target_tx_id)

    net = Network(
        height="680px",
        width="100%",
        bgcolor="#0F172A",
        font_color="#F8FAFC",
        directed=True,
        cdn_resources=cdn_resources,
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, damping=0.95)

    # Add nodes to PyVis
    for node_id, attrs in g.nodes(data=True):
        entity_type = attrs.get("entity_type", "user")
        is_target = attrs.get("is_target", False)

        base_color = NODE_COLORS.get(entity_type, DEFAULT_NODE_COLOR)
        size = 32 if is_target else 20
        border_width = 4 if is_target else 1
        border_color = "#E11D48" if is_target else base_color

        net.add_node(
            node_id,
            label=attrs.get("label", node_id),
            title=attrs.get("title", node_id),
            color={
                "background": base_color,
                "border": border_color,
                "highlight": {"background": "#F43F5E", "border": "#FFE4E6"},
            },
            shape="dot",
            size=size,
            borderWidth=border_width,
            font={"size": 13, "color": "#FFFFFF", "face": "system-ui, -apple-system, sans-serif"},
        )

    # Add edges to PyVis
    for u, v, _key, e_attrs in g.edges(keys=True, data=True):
        is_target_edge = e_attrs.get("is_target_edge", False)
        relation = e_attrs.get("relation", "rel")

        width: float
        if is_target_edge and "target_transacts" in relation:
            color = "#F43F5E"  # Rose red for target transaction
            width = 5.0
            dashes = False
            label = f"${e_attrs.get('amount', 0.0):,.2f} [TARGET TX]"
        elif is_target_edge:
            color = "#F59E0B"  # Amber for target device/IP link
            width = 3.0
            dashes = True
            label = relation
        else:
            color = "#475569"  # Slate for historical background edges
            width = 1.5
            dashes = "uses" in relation
            label = relation if "uses" in relation else ""

        net.add_edge(
            u,
            v,
            color=color,
            width=width,
            dashes=dashes,
            label=label,
            font={"size": 10, "color": "#CBD5E1"},
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Build the HTML in memory: save_graph() writes with the platform default encoding
    # (cp1252 on Windows), which cannot encode the inlined vis-network bundle.
    raw_html: str = net.generate_html(name=out_file.name)

    # Inject a responsive info overlay card

    # Build Header Card with Reason Codes & Post-Hoc Ground Truth
    reasons_html = ""
    if reason_codes:
        reasons_items = "".join(
            f"<li style='margin-bottom: 6px;'>{sentence}</li>" for sentence in reason_codes
        )
        reasons_html = f"""
        <div style="margin-top: 12px; background: rgba(30, 41, 59, 0.7); padding: 12px 16px; border-radius: 8px; border-left: 4px solid #38BDF8;">
            <div style="font-weight: 600; font-size: 14px; color: #38BDF8; margin-bottom: 6px;">Explanatory Reason Codes (Causal Facts Only):</div>
            <ol style="margin: 0; padding-left: 20px; font-size: 13px; color: #E2E8F0; line-height: 1.5;">
                {reasons_items}
            </ol>
        </div>
        """

    score_badge = ""
    if risk_score is not None:
        status_color = "#EF4444" if risk_score >= threshold else "#10B981"
        flag_status = "FLAGGED FOR REVIEW" if risk_score >= threshold else "APPROVED / LOW RISK"
        score_badge = f"""
        <div style="display: flex; gap: 12px; align-items: center; margin-top: 8px;">
            <span style="background: {status_color}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">{flag_status}</span>
            <span style="font-size: 13px; color: #94A3B8;">{model_name} score: <strong style="color: #F8FAFC;">{risk_score * 100:.1f}%</strong> (Threshold: {threshold * 100:.1f}%)</span>
        </div>
        """

    post_hoc_banner = ""
    if is_fraud_ground_truth is not None or post_hoc_fraud_type is not None:
        gt_label = "FRAUDULENT" if is_fraud_ground_truth else "LEGITIMATE"
        gt_color = "#E11D48" if is_fraud_ground_truth else "#059669"
        typology_str = (
            f" | Typology: <strong>{post_hoc_fraud_type}</strong>" if post_hoc_fraud_type else ""
        )
        post_hoc_banner = f"""
        <div style="margin-top: 10px; background: rgba(15, 23, 42, 0.85); padding: 8px 12px; border-radius: 6px; border: 1px dashed {gt_color}; font-size: 12px; color: #CBD5E1;">
            <strong style="color: {gt_color};">[POST-HOC AUDIT ONLY - NOT VISIBLE TO MODEL]</strong> Ground Truth: <strong style="color: {gt_color};">{gt_label}</strong>{typology_str}
        </div>
        """

    overlay_html = f"""
    <div style="position: absolute; top: 16px; left: 16px; max-width: 650px; z-index: 1000; font-family: system-ui, -apple-system, sans-serif; background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(8px); padding: 18px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); border: 1px solid #334155;">
        <h2 style="margin: 0; font-size: 18px; color: #F8FAFC;">Transaction Causal Neighborhood: <code>{target_tx_id}</code></h2>
        {score_badge}
        {reasons_html}
        {post_hoc_banner}
        <div style="margin-top: 12px; display: flex; gap: 14px; font-size: 11px; color: #94A3B8;">
            <span><strong style="color: #38BDF8;">●</strong> User</span>
            <span><strong style="color: #A855F7;">●</strong> Merchant</span>
            <span><strong style="color: #10B981;">●</strong> Device</span>
            <span><strong style="color: #FB923C;">●</strong> IP Address</span>
            <span><strong style="color: #F43F5E;">▬</strong> Target Tx</span>
        </div>
    </div>
    """

    # Inject overlay into body
    injected_html = raw_html.replace("<body>", f"<body>\n{overlay_html}")
    out_file.write_text(injected_html, encoding="utf-8")

    return injected_html


def render_neighborhood_png(
    df: pd.DataFrame,
    target_tx_id: str,
    output_path: str,
    model_name: str,
    risk_score: float,
    threshold: float,
    post_hoc_fraud_type: str | None = None,
    width_in: float = 8.0,
    height_in: float = 5.5,
    dpi: int = 110,
    layout_seed: int = 42,
) -> Path:
    """Render the 2-hop causal neighborhood of one transaction as a static PNG.

    The fraud label is drawn only as a separate, explicitly marked post-hoc overlay; it is
    never an input to the model or to the graph extraction.

    Args:
        df: Transaction dataset.
        target_tx_id: Target transaction ID.
        output_path: Destination PNG path.
        model_name: Model whose score is shown.
        risk_score: That model's score for the transaction.
        threshold: That model's validation-selected threshold.
        post_hoc_fraud_type: Optional ground-truth typology for the post-hoc overlay.
        width_in: Figure width in inches.
        height_in: Figure height in inches.
        dpi: Output resolution.
        layout_seed: Seed for the spring layout (reproducible picture).

    Returns:
        Path of the written PNG.
    """
    g = extract_2hop_causal_subgraph(df, target_tx_id)
    simple = nx.Graph(g)  # collapse parallel edges for layout and drawing
    pos = nx.spring_layout(simple, seed=layout_seed)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    target_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("is_target_edge")]
    other_edges = [
        e for e in simple.edges() if e not in target_edges and e[::-1] not in target_edges
    ]
    nx.draw_networkx_edges(
        simple, pos, edgelist=other_edges, ax=ax, edge_color="#94A3B8", width=1.0
    )
    nx.draw_networkx_edges(
        simple, pos, edgelist=target_edges, ax=ax, edge_color=TARGET_COLOR, width=2.5
    )
    nodes = list(simple.nodes(data=True))
    nx.draw_networkx_nodes(
        simple,
        pos,
        nodelist=[n for n, _ in nodes],
        node_color=[
            NODE_COLORS.get(d.get("entity_type", ""), DEFAULT_NODE_COLOR) for _, d in nodes
        ],
        node_size=[420 if d.get("is_target") else 160 for _, d in nodes],
        edgecolors=[TARGET_COLOR if d.get("is_target") else "white" for _, d in nodes],
        linewidths=[2.0 if d.get("is_target") else 0.5 for _, d in nodes],
        ax=ax,
    )
    label_offset = 0.09  # place target labels below their markers (layout units)
    nx.draw_networkx_labels(
        simple,
        {n: (x, y - label_offset) for n, (x, y) in pos.items()},
        labels={n: d.get("label", n) for n, d in nodes if d.get("is_target")},
        font_size=7,
        ax=ax,
    )

    decision = "flagged" if risk_score >= threshold else "not flagged"
    ax.set_title(
        f"Transaction {target_tx_id}: {model_name}\n"
        f"score {risk_score:.4f}, threshold {threshold:.4f} ({decision}). "
        "Causal 2-hop neighborhood: only transactions before this one.",
        fontsize=9,
    )
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t)
        for t, c in NODE_COLORS.items()
    ] + [Line2D([0], [0], color=TARGET_COLOR, lw=2.5, label="target transaction")]
    ax.legend(handles=legend, loc="upper right", fontsize=7, frameon=False)
    if post_hoc_fraud_type is not None:
        ax.text(
            0.01,
            0.01,
            f"POST-HOC AUDIT ONLY, not a model input: ground truth = {post_hoc_fraud_type}",
            transform=ax.transAxes,
            fontsize=7,
            color=TARGET_COLOR,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": TARGET_COLOR},
        )
    ax.set_axis_off()
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
