"""Generate fake_vlq.py from an IQM VLQ calibration JSON export.

Usage:
    python generate_fake_vlq.py <calibration.json> [-o fake_vlq.py]

The JSON file is the observation-set export from the VLQ device (as produced by
IQM's calibration pipeline). This script reads it, converts units and fidelities
to the shapes expected by IQMErrorProfile, and writes a fake_vlq.py module that
returns a FakeVLQ() backend matching the calibration snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

NUM_QUBITS = 24
QUBITS = [f"QB{i}" for i in range(1, NUM_QUBITS + 1)]
RESONATOR_FAKE = "CR1"       # label used in fake_vlq.py
RESONATOR_JSON = "COMPR1"    # label used in the calibration JSON

# Gate durations (ns) — not present in the JSON, kept as constants.
GATE_DURATIONS_SINGLE = {"prx": 32.0}
GATE_DURATIONS_TWO = {"cz": 88.0, "move": 64.0}


def load_observations(path: Path) -> tuple[dict[str, float], str]:
    """Return a flat {dut_field: value} map plus the snapshot timestamp."""
    with path.open() as f:
        data = json.load(f)
    obs_map = {o["dut_field"]: o["value"] for o in data["observations"] if not o.get("invalid")}
    ts = data.get("created_timestamp") or data.get("end_timestamp") or ""
    return obs_map, ts


def require(obs: dict[str, float], field: str) -> float:
    if field not in obs:
        raise KeyError(f"missing calibration field: {field}")
    return obs[field]


# Decimal places used when formatting generated dictionaries.
T_NS_DECIMALS = 1           # T1/T2 in ns
PRX_DECIMALS = 5            # single-qubit depolarizing parameter
TWO_Q_DECIMALS = 6          # cz / move depolarizing parameters
READOUT_DECIMALS = 4        # readout misclassification probability


def extract_t1(obs: dict[str, float]) -> dict[str, float]:
    """T1 times (seconds → nanoseconds)."""
    result = {RESONATOR_FAKE: round(require(obs, f"characterization.model.{RESONATOR_JSON}.t1_time") * 1e9, T_NS_DECIMALS)}
    for qb in QUBITS:
        result[qb] = round(require(obs, f"characterization.model.{qb}.t1_time") * 1e9, T_NS_DECIMALS)
    return result


def extract_t2(obs: dict[str, float]) -> dict[str, float]:
    """T2 times (seconds → nanoseconds). Falls back to t2_echo_time per qubit
    when t2_time is absent; the resonator reuses its t1 when it has no t2."""
    cr_t2_key = f"characterization.model.{RESONATOR_JSON}.t2_time"
    cr_val = obs.get(cr_t2_key) or obs[f"characterization.model.{RESONATOR_JSON}.t1_time"]
    result = {RESONATOR_FAKE: round(cr_val * 1e9, T_NS_DECIMALS)}
    for qb in QUBITS:
        val = obs.get(f"characterization.model.{qb}.t2_time")
        if val is None:
            val = require(obs, f"characterization.model.{qb}.t2_echo_time")
        result[qb] = round(val * 1e9, T_NS_DECIMALS)
    return result


def extract_prx_errors(obs: dict[str, float]) -> dict[str, float]:
    """Single-qubit prx depolarizing error = 1 - RB fidelity."""
    return {
        qb: round(1.0 - require(obs, f"metrics.rb.prx.drag_crf.{qb}.fidelity:par=d1"), PRX_DECIMALS)
        for qb in QUBITS
    }


def extract_cz_errors(obs: dict[str, float]) -> dict[tuple[str, str], float]:
    """CZ depolarizing error = 1 - IRB fidelity, keyed by (QB, CR1)."""
    return {
        (qb, RESONATOR_FAKE): round(
            1.0 - require(obs, f"metrics.irb.cz.crf_crf.{qb}__{RESONATOR_JSON}.fidelity"),
            TWO_Q_DECIMALS,
        )
        for qb in QUBITS
    }


def extract_move_errors(obs: dict[str, float]) -> dict[tuple[str, str], float]:
    """MOVE depolarizing error per gate = (1 - F_pair) / 2, since the IRB
    metric is reported with n_interleaved_gates=2."""
    return {
        (qb, RESONATOR_FAKE): round(
            (
                1.0
                - require(
                    obs,
                    f"metrics.irb.move.crf_crf.{qb}__{RESONATOR_JSON}"
                    ".fidelity:n_interleaved_gates=2",
                )
            )
            / 2.0,
            TWO_Q_DECIMALS,
        )
        for qb in QUBITS
    }


def extract_readout_errors(obs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Readout misclassification per qubit from ssro.measure.constant."""
    result = {}
    for qb in QUBITS:
        err01 = require(obs, f"metrics.ssro.measure.constant.{qb}.error_0_to_1")
        err10 = require(obs, f"metrics.ssro.measure.constant.{qb}.error_1_to_0")
        result[qb] = {
            "0": round(err01, READOUT_DECIMALS),
            "1": round(err10, READOUT_DECIMALS),
        }
    return result


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _fmt_scalar_dict(d: dict, indent: int) -> str:
    pad = " " * indent
    lines = [f'{pad}"{k}": {v!r},' for k, v in d.items()]
    return "\n".join(lines)


def _fmt_readout_dict(d: dict[str, dict[str, float]], indent: int) -> str:
    pad = " " * indent
    lines = [f'{pad}"{qb}": {{"0": {v["0"]!r}, "1": {v["1"]!r}}},' for qb, v in d.items()]
    return "\n".join(lines)


def _fmt_pair_dict(d: dict[tuple[str, str], float], indent: int) -> str:
    pad = " " * indent
    lines = [f'{pad}("{a}", "{b}"): {v!r},' for (a, b), v in d.items()]
    return "\n".join(lines)


TEMPLATE = '''# {date_comment}
# Generated from calibration file: {calibration_filename}
#
# Copyright 2022-2025 Qiskit on IQM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fake backend for IQM's 24-qubit VLQ architecture."""

from iqm.iqm_client import StaticQuantumArchitecture
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend


def FakeVLQ() -> IQMFakeBackend:
    """Return IQMFakeBackend instance representing IQM's VLQ architecture."""
    architecture = StaticQuantumArchitecture(
        dut_label="FakeVLQ",
        qubits=[{qubit_list}],
        computational_resonators=["{resonator}"],
        connectivity=[
{connectivity}
        ],
    )
    error_profile = IQMErrorProfile(
        t1s={{
{t1s}
        }},
        t2s={{
{t2s}
        }},
        single_qubit_gate_depolarizing_error_parameters={{
            "prx": {{
{prx}
            }}
        }},
        two_qubit_gate_depolarizing_error_parameters={{
            "cz": {{
{cz}
            }},
            "move": {{
{move}
            }},
        }},
        single_qubit_gate_durations={single_durations!r},
        two_qubit_gate_durations={two_durations!r},
        readout_errors={{
{readout}
        }},
        name="VLQ",
    )

    return IQMFakeBackend(architecture, error_profile, name="FakeVLQBackend")
'''


def render(
    t1s: dict,
    t2s: dict,
    prx: dict,
    cz: dict,
    move: dict,
    readout: dict,
    snapshot_ts: str,
    calibration_filename: str,
) -> str:
    try:
        dt = datetime.fromisoformat(snapshot_ts.replace("Z", "+00:00"))
        date_comment = dt.astimezone(timezone.utc).strftime("%B %d, %Y calibration data from VLQ")
    except (ValueError, AttributeError):
        date_comment = "Calibration data from VLQ"

    qubit_list = ", ".join(f'"{q}"' for q in QUBITS)
    connectivity = "\n".join(
        f'            ("{RESONATOR_FAKE}", "{q}"),' for q in QUBITS
    )

    return TEMPLATE.format(
        date_comment=date_comment,
        calibration_filename=calibration_filename,
        qubit_list=qubit_list,
        resonator=RESONATOR_FAKE,
        connectivity=connectivity,
        t1s=_fmt_scalar_dict(t1s, 12),
        t2s=_fmt_scalar_dict(t2s, 12),
        prx=_fmt_scalar_dict(prx, 16),
        cz=_fmt_pair_dict(cz, 16),
        move=_fmt_pair_dict(move, 16),
        single_durations=GATE_DURATIONS_SINGLE,
        two_durations=GATE_DURATIONS_TWO,
        readout=_fmt_readout_dict(readout, 12),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("calibration_json", type=Path, help="Path to the VLQ calibration JSON export")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Destination Python file (default: fake_vlq_YYYY_MM_DD.py "
                        "derived from the calibration snapshot date)")
    args = p.parse_args()

    obs, snapshot_ts = load_observations(args.calibration_json)

    output = args.output
    if output is None:
        try:
            dt = datetime.fromisoformat(snapshot_ts.replace("Z", "+00:00"))
            date_tag = dt.astimezone(timezone.utc).strftime("%Y_%m_%d")
        except (ValueError, AttributeError):
            date_tag = args.calibration_json.stem[:10].replace("-", "_")
        output = args.calibration_json.parent / f"fake_vlq_{date_tag}.py"

    code = render(
        t1s=extract_t1(obs),
        t2s=extract_t2(obs),
        prx=extract_prx_errors(obs),
        cz=extract_cz_errors(obs),
        move=extract_move_errors(obs),
        readout=extract_readout_errors(obs),
        snapshot_ts=snapshot_ts,
        calibration_filename=args.calibration_json.name,
    )
    output.write_text(code)
    print(f"Wrote {output} from {args.calibration_json} (snapshot {snapshot_ts})")


if __name__ == "__main__":
    main()
