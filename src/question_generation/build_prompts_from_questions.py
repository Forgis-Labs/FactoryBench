from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.question_generation.utils.phases import (
    DEFAULT_TASKS_PATH,
    phase_reference_for_question,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_dataset_name(dataset: str) -> str:
    text = str(dataset).strip().lower()
    for prefix in ("cf_", "inter_", "test_", "local_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def machine_description(machine: Dict[str, Any]) -> str:
    model = machine.get("machine_model") or machine.get("model") or ""
    manufacturer = machine.get("manufacturer") or ""
    machine_type = machine.get("machine_type") or machine.get("machine_category") or "industrial machine"

    primary = " ".join(str(x).strip() for x in (manufacturer, model) if str(x).strip())
    if primary:
        return f"{primary} ({machine_type})"
    return str(machine_type)


def resolve_machine_object_for_dataset(dataset: str, machines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    token = normalize_dataset_name(dataset)

    for machine in machines:
        haystack_parts = [
            machine.get("used_in_paper_for"),
            machine.get("machine_model"),
            machine.get("model"),
            machine.get("machine_type"),
            machine.get("machine_category"),
            machine.get("purpose"),
            machine.get("data_source"),
        ]
        haystack = " ".join(str(x).lower() for x in haystack_parts if x)
        if token and token in haystack:
            return machine

    if token == "aursad":
        for machine in machines:
            if str(machine.get("machine_model", "")).lower() == "ur3e":
                return machine

    if token == "vorausad":
        for machine in machines:
            if str(machine.get("machine_model", "")).lower() == "ur5":
                return machine

    if machines:
        return machines[0]
    return None


def load_dataset_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        raw = load_json(path)
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        item["dataset_id"]: item
        for item in raw
        if isinstance(item, dict) and "dataset_id" in item
    }


def resolve_gripper_for_dataset(
    dataset: str,
    grippers: List[Dict[str, Any]],
    dataset_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not grippers:
        return None

    entry = dataset_index.get(str(dataset).strip()) or {}
    gripper_id = entry.get("gripper_id")
    if gripper_id is not None:
        for gripper in grippers:
            if gripper.get("gripper_id") == gripper_id:
                return gripper

    token = normalize_dataset_name(dataset)
    if token == "aursad":
        for gripper in grippers:
            if str(gripper.get("gripper_model", "")).lower() == "screwdriver":
                return gripper
    return None


def gripper_details_block(gripper: Optional[Dict[str, Any]]) -> str:
    if not isinstance(gripper, dict):
        return ""

    manufacturer = str(gripper.get("manufacturer") or "").strip()
    model = str(gripper.get("gripper_model") or "").strip()
    gripper_type = str(gripper.get("gripper_type") or "end-of-arm tool").strip()
    actuation = str(gripper.get("actuation") or "").strip()

    equipment_note = str(gripper.get("equipment_note") or "").strip()

    if model:
        intro = f"{manufacturer} {model}".strip()
        details: List[str] = []
        type_lower = gripper_type.lower()
        if type_lower and type_lower not in intro.lower():
            details.append(type_lower)
    else:
        intro = gripper_type or manufacturer or "gripper"
        details = []
        if equipment_note:
            details.append(equipment_note)
        elif manufacturer and manufacturer.lower() not in intro.lower():
            details.append(manufacturer)

    torque = gripper.get("torque_range_Nm")
    if isinstance(torque, list) and len(torque) == 2:
        details.append(f"{torque[0]}\u2013{torque[1]} Nm torque range")

    grip_force = gripper.get("grip_force_range_N")
    if isinstance(grip_force, list) and len(grip_force) == 2:
        details.append(f"{grip_force[0]}\u2013{grip_force[1]} N grip force")

    screw_range = gripper.get("screw_size_range")
    if isinstance(screw_range, list) and len(screw_range) == 2:
        details.append(f"supporting {screw_range[0]}\u2013{screw_range[1]} screws")

    if details:
        return f"{intro} ({', '.join(details)})"
    return intro


def _indefinite_article(word: str) -> str:
    first = word.strip()[:1].lower()
    return "an" if first in {"a", "e", "i", "o", "u"} else "a"


def machine_details_block(machine: Optional[Dict[str, Any]]) -> str:
    if not isinstance(machine, dict):
        return "industrial machine"

    manufacturer = str(machine.get("manufacturer") or "").strip()
    model = str(machine.get("machine_model") or machine.get("model") or "").strip()
    machine_type = str(machine.get("machine_type") or machine.get("machine_category") or "industrial machine").strip()
    series = str(machine.get("series") or "").strip()

    if manufacturer and model:
        intro = f"{manufacturer} {model}, a {machine_type.lower()}"
    elif model:
        intro = f"{model}, a {machine_type.lower()}"
    else:
        intro = machine_description(machine)

    if series:
        series_lower = series.lower()
        if "series" in series_lower:
            intro += f" from the {series}"
        else:
            intro += f" from the {series} series"

    details: List[str] = []

    payload = machine.get("payload_kg")
    if payload is not None:
        details.append(f"{payload} kg payload")

    dof = machine.get("degrees_of_freedom")
    if dof is not None:
        details.append(f"{dof} degrees of freedom")

    interfaces = machine.get("control_interfaces")
    if isinstance(interfaces, list) and interfaces:
        iface_text = ", ".join(str(x) for x in interfaces[:2])
        details.append(f"controlled via {iface_text}")

    applications = machine.get("typical_applications")
    if isinstance(applications, list) and applications:
        app_text = ", ".join(str(x) for x in applications[:2])
        details.append(f"typically used for {app_text}")

    if details:
        return f"{intro} with " + ", ".join(details)
    return intro


def format_context_time_series(context: Dict[str, Any]) -> str:
    ts = context.get("time_series")
    if isinstance(ts, list):
        return "\n".join(str(x) for x in ts)
    if ts is None:
        return ""
    return str(ts)


def format_acronym_mapping(context: Dict[str, Any]) -> str:
    ts_format = context.get("time_series_format")
    if not isinstance(ts_format, dict):
        return ""

    mapping = ts_format.get("acronym_mapping")
    if not isinstance(mapping, dict) or not mapping:
        return ""

    pairs = [f"{acro}={full}" for acro, full in sorted(mapping.items(), key=lambda kv: str(kv[0]))]
    return "Acronym mapping: " + ", ".join(pairs)


def format_options(options: Any) -> str:
    if isinstance(options, dict) and options:
        keys = sorted(options.keys(), key=lambda k: str(k))
        lines = [f"{k}. {options[k]}" for k in keys]
        return "\n".join(lines)
    if isinstance(options, list) and options:
        lines = [f"{i + 1}. {v}" for i, v in enumerate(options)]
        return "\n".join(lines)
    return ""


def build_prompt(
    question_item: Dict[str, Any],
    machines: List[Dict[str, Any]],
    grippers: Optional[List[Dict[str, Any]]] = None,
    dataset_index: Optional[Dict[str, Dict[str, Any]]] = None,
    episodes_root: Optional[Path] = None,
    tasks_path: Optional[Path] = None,
) -> str:
    provenance = question_item.get("provenance") if isinstance(question_item.get("provenance"), dict) else {}
    dataset = str(provenance.get("dataset", ""))
    hides = set(question_item.get("hides", []))

    machine_sentence = ""
    if "robot" not in hides:
        machine_obj = resolve_machine_object_for_dataset(dataset, machines)
        machine_info = machine_details_block(machine_obj)
        machine_sentence = f"The following sensor data comes from {machine_info}."
    if "gripper" not in hides:
        gripper_obj = resolve_gripper_for_dataset(dataset, grippers or [], dataset_index or {})
        gripper_info = gripper_details_block(gripper_obj)
        if gripper_info:
            article = _indefinite_article(gripper_info)
            if machine_sentence:
                machine_sentence += f" It is equipped with {article} {gripper_info}."
            else:
                machine_sentence = f"The robot is equipped with {article} {gripper_info}."

    context = question_item.get("context") if isinstance(question_item.get("context"), dict) else {}
    ts_text = format_context_time_series(context)
    mapping_text = format_acronym_mapping(context)
    question_text = str(question_item.get("question", "")).strip()
    options_text = format_options(question_item.get("options"))

    # Phase legend. Only attached when the question or the context actually
    # involves phases, so prompts with no phase content are unchanged. States
    # what each phase means and the order they run in; never which timesteps of
    # this episode belong to which phase, which is what L2 template 6 asks the
    # model to work out.
    phase_text = phase_reference_for_question(
        question_item,
        episodes_root if episodes_root is not None else Path("data") / "normalized_episodes",
        tasks_path if tasks_path is not None else DEFAULT_TASKS_PATH,
        dataset_index=dataset_index,
    )

    header_parts = [p for p in [machine_sentence, phase_text, mapping_text] if p]
    header = "\n".join(header_parts)
    prompt = f"{header}\n{ts_text}\nQuestion: {question_text}" if header else f"{ts_text}\nQuestion: {question_text}"
    if options_text:
        prompt += f"\nHere are the options:\n{options_text}"
    return prompt


def is_question_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "question" in payload and "context" in payload


def iter_question_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def detect_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "labelling" / "machines.json").exists() and (candidate / "src").exists():
            return candidate
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompt-only JSON files from a question folder.")
    repo_root = detect_repo_root(Path(__file__).resolve())

    parser.add_argument("--input", type=Path, required=True, help="Input question folder")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for prompt JSON files")
    parser.add_argument(
        "--machines",
        type=Path,
        default=repo_root / "data" / "labelling" / "machines.json",
        help="Path to machines.json",
    )
    parser.add_argument(
        "--grippers",
        type=Path,
        default=repo_root / "data" / "labelling" / "grippers.json",
        help="Path to grippers.json (optional; skipped if missing)",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=repo_root / "data" / "labelling" / "dataset.json",
        help="Path to dataset.json (maps dataset_id -> gripper_id/machine_id)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=repo_root / "data" / "labelling" / "tasks.json",
        help="Path to tasks.json (source of the task-phase definitions)",
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=repo_root / "data" / "normalized_episodes",
        help="Root of normalized episodes; used to resolve each question's task "
             "from its *_metadata.json sidecar so the right phase legend is attached.",
    )
    parser.add_argument(
        "--no-phase-reference",
        action="store_true",
        help="Omit the task-phase legend. Only for reproducing the pre-legend "
             "prompts the published numbers were measured on.",
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    machines_path = args.machines.resolve()
    grippers_path = args.grippers.resolve() if args.grippers else None
    dataset_index_path = args.dataset_index.resolve() if args.dataset_index else None

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not machines_path.exists() or not machines_path.is_file():
        raise FileNotFoundError(f"Machines file not found: {machines_path}")

    machines_raw = load_json(machines_path)
    machines = machines_raw if isinstance(machines_raw, list) else []

    grippers: List[Dict[str, Any]] = []
    if grippers_path and grippers_path.exists() and grippers_path.is_file():
        grippers_raw = load_json(grippers_path)
        if isinstance(grippers_raw, list):
            grippers = grippers_raw

    dataset_index = load_dataset_index(dataset_index_path)

    converted = 0
    scanned = 0
    with_phases = 0

    # A sentinel path that cannot resolve makes phase_reference_for_question
    # return "" for every item, which is exactly what --no-phase-reference means.
    tasks_path = Path("\0") if args.no_phase_reference else args.tasks.resolve()
    episodes_root = args.episodes_root.resolve()

    for in_path in iter_question_files(input_dir):
        scanned += 1
        try:
            payload = load_json(in_path)
        except Exception:
            continue

        if not is_question_payload(payload):
            continue

        prompt = build_prompt(
            payload, machines, grippers, dataset_index,
            episodes_root=episodes_root,
            tasks_path=tasks_path,
        )
        if "The task runs through the following phases" in prompt:
            with_phases += 1
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        write_json(out_path, {
            "prompt": prompt,
            "metadata": {"qa_pair_id": payload.get("id")},
        })
        converted += 1

    print(f"Scanned {scanned} JSON files; converted {converted} question files to prompts in {output_dir}")
    if args.no_phase_reference:
        print("Phase legend: disabled (--no-phase-reference)")
    else:
        print(f"Phase legend: attached to {with_phases}/{converted} prompts")


if __name__ == "__main__":
    main()
