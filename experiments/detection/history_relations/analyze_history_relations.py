"""Describe requester/retained history relations, without fitting a detector.

Uses verified compact observations, not weights, labels, triggers or FU. The
existing HRAC chooses ONE coordinate per client in the first half; that same
coordinate is inspected in the second half. All clients and normal controls
receive the same rule. Known roles/targets enter separate diagnostic reports.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.detection.temporal_history import temporal_history as th
from experiments.detection.temporal_history.validate_temporal_history import exported, write_json

VERSION = "history-relations-descriptive-v1"


def request_scope(spec):
    """Explicitly separate the paper's deletion unit from the saved execution.

    This is reporting metadata, not a rule choosing different detectors by name.
    No within-client atoms or request observations are reconstructed from sums.
    """
    if spec["name"].startswith("fuba"):
        paper = "whole Adv-defender client"
        execution = "prospective whole-client deletion; reference client 4"
        distinction = "Client-level boundary is represented, but FU effects are not measured here."
    else:
        paper = "camouflage samples inside one malicious client; poison samples retained"
        execution = ("full6: appended client 5 contains clean client-0 data plus cover; assessed as whole-client deletion"
                     if spec["name"] == "badfu_full6" else "clean5: hypothetical whole-client deletions; no attack")
        distinction = "The full6 boundary is NOT a validation of within-client camouflage-only deletion."
    path = ROOT/spec["evidence"].partition("::")[0]
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        has_requests = "evidence/requests.json" in names
        requests = json.loads(z.read("evidence/requests.json")) if has_requests else None
        n = sum("intra" in r for r in requests["requests"]) if requests else 0
        inventory = json.loads(z.read("evidence/inventory.json"))
        bound = "requests.json" in inventory if has_requests else None
    return dict(paper_deletion_unit=paper, evaluated_deletion_unit=execution, distinction=distinction,
                recorded_history_unit="client update", inter_relation="observed_client_boundary",
                request_file_present=has_requests, request_file_inventory_bound=bound,
                declared_requests=len(requests["requests"]) if requests else 0, intra_payload_count=n,
                intra_status="unobserved_independent_request_effect_and_internal_atoms" if n == 0 else "not_evaluated_by_this_inter_diagnostic",
                intra_score=None, within_client_boundary_status="not_evaluated", joint_verdict=None)


def validate(h, mask):
    h, mask = np.asarray(h, float), np.asarray(mask)
    if (h.ndim != 3 or len(h) < 2 or h.shape[1] < 2 or h.shape[2] < 1
            or mask.shape != h.shape[:2] or mask.dtype.kind != "b"
            or not np.isfinite(h).all() or np.any(h[~mask] != 0)):
        raise ValueError("Finite [round, client, coordinate] and actual boolean participation required")
    return h, mask


def select_prefix(h, mask, cutoff):
    """Existing HRAC at cutoff; no future observations or role labels used."""
    h, mask = validate(h, mask)
    if not isinstance(cutoff, int) or not 1 <= cutoff < len(h):
        raise ValueError("Cutoff must leave an observed prefix and suffix")
    values = th.release(h[:cutoff].sum(0))
    coordinates = values.argmax(-1)  # deterministic first index for exact ties
    maxima = values[np.arange(h.shape[1]), coordinates]
    ties = (values == maxima[:, None]).sum(-1)
    return coordinates, maxima, ties


def decompose(h, mask, client, partners=None):
    h, mask = validate(h, mask)
    if not isinstance(client, int) or not 0 <= client < h.shape[1]:
        raise ValueError("Invalid requester index")
    ids = [i for i in range(h.shape[1]) if i != client] if partners is None else list(partners)
    if (not ids or len(ids) != len(set(ids)) or client in ids
            or any(not isinstance(i, int) or not 0 <= i < h.shape[1] for i in ids)):
        raise ValueError("Partners must be nonempty, distinct, valid retained clients")
    # Sum the retained set directly, avoiding total-minus-own cancellation.
    return h[:, client], h[:, ids].sum(1), mask[:, client] & mask[:, ids].any(1)


def longest_true(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def relation_summary(x, y, eligible):
    """Descriptive scalars for ONE fixed coordinate, no decision threshold.

    x<0,y>0 is the original positive-release orientation. Both orientations
    are reported. Zero magnitude has no sign. Missed rounds break physical
    streaks; opportunity streaks are explicitly a different quantity.
    """
    x, y, eligible = np.asarray(x, float), np.asarray(y, float), np.asarray(eligible)
    if (x.ndim != 1 or x.shape != y.shape or eligible.shape != x.shape or not len(x)
            or eligible.dtype.kind != "b" or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Finite 1-D matching series and boolean eligibility required")
    forward = eligible & (x < 0) & (y > 0)
    reverse = eligible & (x > 0) & (y < 0)
    n = int(eligible.sum())
    denominator = float((np.abs(x[eligible]) + np.abs(y[eligible])).sum())
    overlap = float(2 * np.minimum(np.maximum(-x[eligible], 0), np.maximum(y[eligible], 0)).sum())
    bilateral = float(2 * np.minimum(np.abs(x[eligible]), np.abs(y[eligible]))[
        (forward | reverse)[eligible]].sum())
    return dict(eligible_rounds=n, forward_rounds=int(forward.sum()), reverse_rounds=int(reverse.sum()),
                forward_fraction=float(forward.sum()/n) if n else None,
                either_fraction=float((forward | reverse).sum()/n) if n else None,
                forward_longest_physical=longest_true(forward),
                forward_longest_opportunities=longest_true(forward[eligible]),
                forward_overlap_fraction=overlap/denominator if denominator else None,
                either_overlap_fraction=bilateral/denominator if denominator else None,
                own_sum=float(x.sum()), retained_sum=float(y.sum()),
                window_cumulative_forward=bool(x.sum() < 0 and y.sum() > 0))


def coordinate_info(e, d):
    views = e["coordinates"]["views"][1:]
    repeats = len(e["coordinates"]["probe_seeds"])
    outputs, remainder = divmod(e["history"].shape[-1], repeats*len(views))
    if remainder:
        raise ValueError("Coordinate metadata does not match dimensions")
    rep, within = divmod(int(d), len(views)*outputs)
    view, output = divmod(within, outputs)
    return dict(index=int(d), probe_seed=e["coordinates"]["probe_seeds"][rep],
                view=views[view], output=int(output))


def sign_counts(x, y):
    return dict(forward=int(((x < 0) & (y > 0)).sum()),
                reverse=int(((x > 0) & (y < 0)).sum()),
                both_positive=int(((x > 0) & (y > 0)).sum()),
                both_negative=int(((x < 0) & (y < 0)).sum()),
                zero=int(((x == 0) | (y == 0)).sum()))


def default_specs():
    fuba = [dict(name="fuba_attack42", evidence=exported("fuba_cifar10_s42"),
                 report_client=4, partners=[0, 1, 2, 3], target=8)]
    fuba += [dict(name=f"fuba_normal{s}", evidence=exported(f"fuba_cifar10_benign_s{s}"),
                  report_client=4, partners=[0, 1, 2, 3], target=8) for s in (42, 43, 44)]
    badfu = exported("badfu_author_s42")
    return fuba + [dict(name="badfu_full6", evidence=badfu+"::full6", report_client=5,
                        partners=[0], target=0),
                   dict(name="badfu_clean5", evidence=badfu+"::clean5", report_client=0,
                        partners=None, target=0)]


def analyze(e, spec):
    h, mask = validate(e["history"], e["mask"])
    cutoff = len(h)//2
    coords, maxima, ties = select_prefix(h, mask, cutoff)
    clients, arrays = [], {}
    for i, d in enumerate(coords):
        x, y, eligible = decompose(h, mask, i)
        x, y = x[:, d], y[:, d]
        row = dict(client=i, coordinate=coordinate_info(e, d), prefix_hrac=float(maxima[i]),
                   identified=bool(maxima[i] > 0), exact_coordinate_ties=int(ties[i]),
                   first_half=relation_summary(x[:cutoff], y[:cutoff], eligible[:cutoff]),
                   second_half=relation_summary(x[cutoff:], y[cutoff:], eligible[cutoff:]),
                   whole=relation_summary(x, y, eligible))
        clients.append(row)
    arrays["fixed_coordinate"] = coords
    arrays["own"] = np.take_along_axis(h, coords[None, :, None], axis=2)[..., 0]
    arrays["retained"] = np.stack([decompose(h, mask, i)[1][:, coords[i]] for i in range(h.shape[1])], axis=1)
    arrays["participation"] = mask
    ref = spec["report_client"]
    x, y, eligible = decompose(h, mask, ref)
    _, partner, partner_eligible = decompose(h, mask, ref, spec["partners"])
    # Actual target is used ONLY here, after all client coordinates were fixed.
    target_dims = np.array([d for d in range(h.shape[-1]) if coordinate_info(e, d)["output"] == spec["target"]])
    arrays["target_own"] = x[:, target_dims]
    arrays["target_retained"] = y[:, target_dims]
    arrays["target_partner"] = partner[:, target_dims]
    target = dict(output=spec["target"], dimensions=target_dims.tolist(),
                  note="Known target/roles are post-hoc diagnostics, never selection inputs.", comparisons={})
    for name, other, active in (("all_retained", y, eligible), ("named_group", partner, partner_eligible)):
        counts = {str(t): sign_counts(x[:t].sum(0)[target_dims], other[:t].sum(0)[target_dims])
                  for t in (cutoff, len(h))}
        xx, yy = x[:, target_dims], other[:, target_dims]
        n = int(active.sum()) * len(target_dims)
        target["comparisons"][name] = dict(cumulative_signs=counts,
            eligible_coordinate_rounds=n,
            forward_coordinate_rounds=int(((xx < 0) & (yy > 0) & active[:, None]).sum()),
            both_positive_coordinate_rounds=int(((xx > 0) & (yy > 0) & active[:, None]).sum()))
    final_scores = th.release(h.sum(0))[ref]
    d_final = int(final_scores.argmax())
    # Full-run selection is included only to explain existing endpoint HRAC.
    end = dict(coordinate=coordinate_info(e, d_final), hrac=float(final_scores[d_final]),
               retrospective_only=True, whole=relation_summary(x[:, d_final], y[:, d_final], eligible))
    d = int(coords[ref])
    role = dict(client=ref, partners=spec["partners"], prefix_coordinate=coordinate_info(e, d),
                whole_vs_group=relation_summary(x[:, d], partner[:, d], partner_eligible),
                second_half_vs_group=relation_summary(x[cutoff:, d], partner[cutoff:, d], partner_eligible[cutoff:]),
                final_components=dict(requester=float(x[:, d].sum()), named_group=float(partner[:, d].sum()),
                                      other_retained=float((y[:, d]-partner[:, d]).sum())))
    return dict(name=spec["name"], source=spec["evidence"], inventory_sha256=e["inventory_sha256"],
                response_sha256=e["identity"], benign=e["benign"], stratum=e["stratum"], cutoff=cutoff,
                clients=clients, prefix_top_clients=np.flatnonzero(maxima == maxima.max()).tolist(),
                role_diagnostic=role, target_diagnostic=target, endpoint_diagnostic=end), arrays


def plot_relations(episodes, rows, specs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    for family, indices in (("fuba", range(4)), ("badfu", range(4, 6))):
        source = indices[0]
        d = rows[source]["role_diagnostic"]["prefix_coordinate"]["index"]
        if any(episodes[i]["coordinates"] != episodes[source]["coordinates"] for i in indices):
            raise ValueError("Cannot transfer a coordinate between different probe systems")
        fig, axs = plt.subplots(len(indices), 2, figsize=(13, 3*len(indices)), squeeze=False)
        for axes, i in zip(axs, indices):
            e, spec, row = episodes[i], specs[i], rows[i]
            x, y, active = decompose(e["history"], e["mask"], spec["report_client"])
            _, p, _ = decompose(e["history"], e["mask"], spec["report_client"], spec["partners"])
            rounds = np.arange(1, len(x)+1)
            own, rest = x[:, d], y[:, d]
            stat = relation_summary(own[row["cutoff"]:], rest[row["cutoff"]:], active[row["cutoff"]:])
            axes[0].plot(rounds, own, color="#b63f37", lw=1.2, label=f"client {spec['report_client']}")
            axes[0].plot(rounds, rest, color="#256c9f", lw=1.1, label="all retained")
            axes[0].scatter(rounds[~e["mask"][:, spec["report_client"]]],
                            np.zeros((~e["mask"][:, spec["report_client"]]).sum()),
                            marker="x", color="#999999", s=13, label="requester absent")
            axes[0].set_title(f"{row['name']} | new round contribution")
            axes[1].plot(rounds, own.cumsum(), color="#b63f37", label="requester history")
            axes[1].plot(rounds, rest.cumsum(), color="#256c9f", label="retained history")
            axes[1].plot(rounds, (own+rest).cumsum(), color="#333333", ls=":", label="sum (proxy)")
            if not e["benign"]:
                axes[1].plot(rounds, p[:, d].cumsum(), color="#a77718", ls="--", label=f"known group {spec['partners']}")
            axes[1].set_title(f"Fixed coordinate | later opposition {stat['forward_rounds']}/{stat['eligible_rounds']} active rounds")
            for ax in axes:
                ax.axhline(0, color="#777777", lw=.6)
                ax.axvline(row["cutoff"]+.5, color="#555555", ls="--", lw=.7)
                ax.set_xlabel("FL round")
                ax.set_ylabel("paired centered-logit response")
                ax.grid(alpha=.15)
                ax.legend(fontsize=7, loc="best")
        info = coordinate_info(episodes[source], d)
        fig.suptitle(f"{family.upper()}: CLIENT-LEVEL boundary; coordinate fixed at round {rows[source]['cutoff']}\n"
                     f"probe {info['probe_seed']} / {info['view']} / output {info['output']}; transferred to controls", fontsize=12)
        fig.tight_layout(rect=(0, 0.025, 1, .94))
        fig.text(.5, .008, "No within-client deletion tested. BadFU full6 removes appended clean+cover client; clean5 is unmatched.", ha="center", fontsize=8)
        fig.savefig(out / f"{family}_fixed_relation.png", dpi=160)
        plt.close(fig)

    # ALL 36 target coordinates; no best-pattern selection. Cumulative signs.
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    colors = ["#d6d6d6", "#237f8b", "#c75850", "#d6ae4b", "#756a99"]
    cmap, norm = ListedColormap(colors), BoundaryNorm(np.arange(-.5, 5), 5)
    for axes, i in zip(axs, (0, 4)):
        e, row, spec = episodes[i], rows[i], specs[i]
        x, y, _ = decompose(e["history"], e["mask"], spec["report_client"])
        _, p, _ = decompose(e["history"], e["mask"], spec["report_client"], spec["partners"])
        dims = row["target_diagnostic"]["dimensions"]
        a = x[:, dims].cumsum(0)
        for ax, b, name in zip(axes, (y[:, dims].cumsum(0), p[:, dims].cumsum(0)), ("all retained", "known poison group")):
            states = np.zeros_like(a, int)
            states[(a < 0) & (b > 0)] = 1
            states[(a > 0) & (b > 0)] = 2
            states[(a > 0) & (b < 0)] = 3
            states[(a < 0) & (b < 0)] = 4
            im = ax.imshow(states.T, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                           extent=(.5, len(a)+.5, -.5, len(dims)-.5), interpolation="nearest")
            ax.axvline(row["cutoff"]+.5, color="#111111", lw=.8, ls="--")
            ax.set_title(f"{row['name']} | output {spec['target']} vs {name}")
            ax.set_xlabel("FL round")
            ax.set_ylabel("fixed probe-pattern coordinate (all 36)")
    fig.suptitle("CLIENT-LEVEL relation: requester < 0 and retained > 0?\nKnown target/roles: explanation only. Within-client deletion is NOT tested.", fontsize=12)
    fig.tight_layout(rect=(0, .13, 1, .92))
    bar = fig.colorbar(im, cax=fig.add_axes((.12, .045, .76, .025)), orientation="horizontal", ticks=range(5))
    bar.ax.set_xticklabels(["zero", "requester - / rest +", "both +", "requester + / rest -", "both -"], fontsize=9)
    fig.savefig(out / "target_relation_signs.png", dpi=160)
    plt.close(fig)


def execute(out=None):
    specs = default_specs()
    episodes = []
    for spec in specs:
        path, sep, phase = spec["evidence"].partition("::")
        episodes.append(th.load_evidence(str(ROOT/path) + (sep+phase if sep else "")))
    protocol = dict(version=VERSION, cutoff="floor(planned horizon / 2)",
        selection="existing cumulative HRAC at cutoff, per client over all 360 coordinates; exact ties use first index",
        inspection="same fixed coordinate in later rounds; all clients; missing participation is not sign evidence",
        controls="FUBA three attack-disabled runs; BadFU clean5 is an unmatched diagnostic control",
        targets="known identities/outputs only in explicitly marked post-hoc diagnostics",
        inference="descriptive; no fitted detector, alarm, FPR, p-value, causal FU estimate or novelty claim",
        observation="paired public-pattern mean centered-logit responses at changing pre-round models; finite nonlinear effects are not additive model effects",
        scope="client-level inter relation only; retain existing intra feasibility formula, do not infer intra success/failure",
        specs=specs, code_sha256={p.name: th.digest(p.read_bytes()) for p in
            (Path(__file__), Path(th.__file__), ROOT/"experiments/detection/temporal_history/validate_temporal_history.py")})
    signature = th.digest(json.dumps(dict(protocol=protocol, inventories=[e["inventory_sha256"] for e in episodes]), sort_keys=True).encode())
    out = Path(out) if out is not None else ROOT/"experiments/detection/history_relations/results"/signature[:16]
    if out.exists():
        old = json.loads((out/"summary.json").read_text(encoding="utf-8"))
        if old.get("signature") != signature or old.get("status") != "complete":
            raise ValueError("Different/incomplete output exists; use a new --out")
        for name, sha in json.loads((out/"inventory.json").read_text()).items():
            if th.digest((out/name).read_bytes()) != sha:
                raise ValueError("Output hash mismatch; preserve it and use a new --out")
        print(f"Verified existing relation analysis: {out}")
        return out
    rows, arrays = [], {}
    for e, spec in zip(episodes, specs):
        row, a = analyze(e, spec)
        row["request_scope"] = request_scope(spec)
        rows.append(row)
        arrays.update({spec["name"]+"_"+k: v for k, v in a.items()})
    # Also report the exact common-coordinate comparison used in the plots.
    # Control-specific prefix selections above remain the primary comparison.
    for source, indices in ((0, range(4)), (4, range(4, 6))):
        d = rows[source]["role_diagnostic"]["prefix_coordinate"]["index"]
        for i in indices:
            e, spec, row = episodes[i], specs[i], rows[i]
            if e["coordinates"] != episodes[source]["coordinates"]:
                raise ValueError("Transferred coordinate systems differ")
            x, y, active = decompose(e["history"], e["mask"], spec["report_client"])
            cut = row["cutoff"]
            row["transferred_coordinate_diagnostic"] = dict(source_run=rows[source]["name"],
                coordinate=coordinate_info(e, d), client=spec["report_client"],
                whole=relation_summary(x[:, d], y[:, d], active),
                second_half=relation_summary(x[cut:, d], y[cut:, d], active[cut:]))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", protocol)
    write_json(out/"summary.json", dict(status="complete", signature=signature, runs=rows))
    np.savez_compressed(out/"series.npz", **arrays)
    buf = io.StringIO(newline="")
    fields = ["run", "client", "coordinate", "output", "prefix_hrac", "identified", "later_eligible",
              "later_forward", "later_fraction", "later_physical_streak", "later_opportunity_streak", "later_overlap_fraction"]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        for c in r["clients"]:
            s = c["second_half"]
            writer.writerow(dict(run=r["name"], client=c["client"], coordinate=c["coordinate"]["index"],
                output=c["coordinate"]["output"], prefix_hrac=c["prefix_hrac"], identified=c["identified"],
                later_eligible=s["eligible_rounds"], later_forward=s["forward_rounds"], later_fraction=s["forward_fraction"],
                later_physical_streak=s["forward_longest_physical"], later_opportunity_streak=s["forward_longest_opportunities"],
                later_overlap_fraction=s["forward_overlap_fraction"]))
    (out/"fixed_coordinate_clients.csv").write_bytes(buf.getvalue().encode("utf-8"))
    plot_relations(episodes, rows, specs, out)
    write_json(out/"inventory.json", {p.name: th.digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory; default is content-addressed")
    execute(parser.parse_args().out)
