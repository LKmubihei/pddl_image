"""PDDL Domain Compiler
======================
Dynamically compiles a PDDL domain + typed objects into all model inputs:
- types, predicates with full signatures
- canonical ground atom vocabulary
- schema info for predicate queries (name, arity, arg types, action roles, gloss)
- action semantics masks (precondition_mask, effect_delta)

This version uses unified_planning to parse the PDDL domain so we can
recover lifted preconditions and conditional effects more faithfully.
"""
from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import Any

from unified_planning.io import PDDLReader
from unified_planning.model import Object


def _canonical_name(name: str) -> str:
    """Map UP lower-case names back to the blocksworld camel-case names."""
    mapping = {
        "moveblock": "moveBlock",
        "incolumn": "inColumn",
        "rightof": "rightOf",
        "leftof": "leftOf",
    }
    return mapping.get(name.lower(), name)


def _canonical_arg_name(raw: str, object_name_map: dict[str, str] | None = None) -> str:
    """Return the compile-time object spelling for a grounded UP argument."""
    key = raw.replace(" ", "")
    if object_name_map is not None:
        mapped = object_name_map.get(key.lower())
        if mapped is not None:
            return mapped
    return key.upper()


def _infer_atom_string(
    expr: Any,
    object_name_map: dict[str, str] | None = None,
) -> str | None:
    """Convert a unified_planning expression into a canonical atom string."""
    if expr is None:
        return None
    if hasattr(expr, "is_true") and expr.is_true():
        return "(true)"
    if hasattr(expr, "is_false") and expr.is_false():
        return "(false)"
    if hasattr(expr, "is_not") and expr.is_not():
        args = list(expr.args)
        if len(args) == 1:
            inner = _infer_atom_string(args[0], object_name_map)
            if inner is None:
                return None
            return f"(not {inner})"
    if hasattr(expr, "is_and") and expr.is_and():
        parts = []
        for a in expr.args:
            s = _infer_atom_string(a, object_name_map)
            if s is not None:
                parts.append(s)
        if not parts:
            return None
        return "(and " + " ".join(parts) + ")"
    if hasattr(expr, "is_or") and expr.is_or():
        parts = []
        for a in expr.args:
            s = _infer_atom_string(a, object_name_map)
            if s is not None:
                parts.append(s)
        if not parts:
            return None
        return "(or " + " ".join(parts) + ")"
    if hasattr(expr, "is_fluent_exp") and expr.is_fluent_exp():
        fluent = expr.fluent()
        args = [_canonical_arg_name(str(a), object_name_map) for a in expr.args]
        name = _canonical_name(fluent.name)
        return f"({name} {' '.join(args)})" if args else f"({name})"
    return None


def _collect_positive_atoms(
    expr: Any,
    object_name_map: dict[str, str] | None = None,
) -> list[str]:
    """Collect positive atom strings from a UP boolean expression."""
    if expr is None:
        return []
    if hasattr(expr, "is_and") and expr.is_and():
        out = []
        for a in expr.args:
            out.extend(_collect_positive_atoms(a, object_name_map))
        return out
    if hasattr(expr, "is_not") and expr.is_not():
        return []
    atom = _infer_atom_string(expr, object_name_map)
    return [atom] if atom is not None else []


def _collect_effect_items(
    effect: Any,
    object_name_map: dict[str, str] | None = None,
) -> list[tuple[str, bool]]:
    """Collect (atom, is_positive) pairs from a UP effect."""
    items: list[tuple[str, bool]] = []
    if effect is None:
        return items
    if getattr(effect, "is_forall", lambda: False)():
        return items
    if getattr(effect, "is_conditional", lambda: False)() and effect.condition is not None:
        # Keep conditional effects, but extract the assigned fluent.
        atom = _infer_atom_string(effect.fluent, object_name_map)
        if atom is not None:
            val = effect.value
            is_positive = hasattr(val, "is_true") and val.is_true()
            items.append((atom, is_positive))
        return items
    atom = _infer_atom_string(effect.fluent, object_name_map)
    if atom is None:
        return items
    val = effect.value
    is_positive = hasattr(val, "is_true") and val.is_true()
    items.append((atom, is_positive))
    return items


def _up_type_name(up_type: Any) -> str:
    return str(up_type).replace(" ", "")


class PDDLDomainCompiler:
    """Compile a PDDL domain file + typed objects into model-ready structures.

    Usage::

        compiler = PDDLDomainCompiler("domain.pddl")
        info = compiler.compile(objects={"block": ["Y","P","R"], "column": ["C1","C2","C3","C4"]})

        # info.types, info.predicate_schemas, info.canonical_atoms,
        # info.action_semantics, etc.
    """

    def __init__(self, domain_path: str):
        self.domain_path = str(domain_path)
        reader = PDDLReader()
        dummy_problem = self._find_dummy_problem(domain_path)
        self.problem = reader.parse_problem(self.domain_path, dummy_problem)
        self.domain_name: str = self.problem.name
        self.types: list[str] = [t.name for t in self.problem.user_types]
        self._domain = self.problem.kind
        self.up_type_objs = {t.name: t for t in self.problem.user_types}

    # ------------------------------------------------------------------
    # Main compile method
    # ------------------------------------------------------------------
    def compile(
        self,
        objects: dict[str, list[str]],
        static_predicates: set[str] | None = None,
    ) -> DomainInfo:
        """Compile domain for a specific set of typed objects.

        Args:
            objects: mapping type_name -> list of object names.
                     e.g. {"block": ["Y","P","R"], "column": ["C1","C2","C3","C4"]}
            static_predicates: predicate names that are static (not predicted).
                               These are excluded from canonical atoms.

        Returns:
            DomainInfo with all compiled structures.
        """
        static = static_predicates or set()

        # 1. Build type index
        type_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.types)}

        # 2. Build object list with types
        all_objects: list[ObjectInfo] = []
        obj_name_to_idx: dict[str, int] = {}
        object_name_map: dict[str, str] = {}
        up_objects = {}
        for type_name, obj_list in objects.items():
            for obj_name in obj_list:
                idx = len(all_objects)
                obj_key = obj_name.lower()
                all_objects.append(ObjectInfo(
                    name=obj_name,
                    type_name=type_name,
                    type_idx=type_to_idx.get(type_name, 0),
                ))
                obj_name_to_idx[obj_key] = idx
                object_name_map[obj_key] = obj_name
                if type_name in self.up_type_objs:
                    up_objects[obj_key] = Object(obj_key, self.up_type_objs[type_name])
        self.up_objects = up_objects
        self.object_name_map = object_name_map

        # 3. Build predicate schemas (dynamic only)
        dynamic_preds = []
        for f in self.problem.fluents:
            if f.name in static:
                continue
            dynamic_preds.append(f)
        predicate_schemas: list[PredicateSchema] = []
        for pred in dynamic_preds:
            name = _canonical_name(pred.name)
            arity = len(pred.signature)
            param_types = [_up_type_name(p.type) for p in pred.signature]
            param_names = [p.name for p in pred.signature]

            # Determine action roles from action definitions
            action_roles = self._get_action_roles(name)

            # Generate natural language gloss
            gloss = self._generate_gloss(name, arity, param_types, param_names)

            predicate_schemas.append(PredicateSchema(
                name=name,
                arity=arity,
                param_types=param_types,
                param_names=param_names,
                action_roles=action_roles,
                gloss=gloss,
            ))

        # Sort by name for canonical ordering
        predicate_schemas.sort(key=lambda s: s.name)

        # 4. Build canonical ground atoms
        canonical_atoms: list[GroundAtom] = []
        for schema in predicate_schemas:
            if schema.arity == 0:
                canonical_atoms.append(GroundAtom(
                    predicate=schema.name,
                    arguments=(),
                    str_repr=f"({schema.name})",
                    predicate_idx=predicate_schemas.index(schema),
                ))
            else:
                param_obj_lists = [objects.get(pt, []) for pt in schema.param_types]
                for combo in product(*param_obj_lists):
                    # Skip self-relations for same-type binary predicates
                    if schema.arity == 2 and schema.param_types[0] == schema.param_types[1]:
                        if combo[0] == combo[1]:
                            continue
                    canonical_atoms.append(GroundAtom(
                        predicate=schema.name,
                        arguments=combo,
                        str_repr=f"({schema.name} {' '.join(combo)})",
                        predicate_idx=predicate_schemas.index(schema),
                    ))

        # 5. Build action semantics from grounded action instances.
        action_semantics = self._build_grounded_action_semantics(
            predicate_schemas, canonical_atoms, objects, static
        )

        # 6. Build object-to-slot mapping
        obj_type_ids = [o.type_idx for o in all_objects]

        return DomainInfo(
            domain_name=self.domain_name,
            types=self.types,
            type_to_idx=type_to_idx,
            objects=all_objects,
            obj_name_to_idx=obj_name_to_idx,
            predicate_schemas=predicate_schemas,
            canonical_atoms=canonical_atoms,
            action_semantics=action_semantics,
            obj_type_ids=obj_type_ids,
            static_predicates=static,
            n_canonical=len(canonical_atoms),
        )

    # ------------------------------------------------------------------
    # Action role analysis
    # ------------------------------------------------------------------
    def _get_action_roles(self, predicate_name: str) -> list[str]:
        """Determine what roles a predicate plays across all actions."""
        roles = set()
        for action in self.problem.actions:
            # Check precondition
            if any(predicate_name.lower() in str(pre).lower() for pre in action.preconditions):
                roles.add("precondition")
            # Check effect
            if any(predicate_name.lower() in str(eff).lower() for eff in action.effects):
                roles.add("effect")
        return sorted(roles)

    # ------------------------------------------------------------------
    # Action semantics masks
    # ------------------------------------------------------------------
    def _build_grounded_action_semantics(
        self,
        predicate_schemas: list[PredicateSchema],
        canonical_atoms: list[GroundAtom],
        objects: dict[str, list[str]],
        static: set[str],
    ) -> list[ActionSemantics]:
        """Build grounded precondition mask and effect delta for each action instance."""
        atom_str_to_idx = {a.str_repr: i for i, a in enumerate(canonical_atoms)}
        n_atoms = len(canonical_atoms)
        semantics_list = []

        for action in self.problem.actions:
            param_names = [p.name for p in action.parameters]
            param_types = [_up_type_name(p.type) for p in action.parameters]
            candidate_lists = [objects.get(t, []) for t in param_types]
            for combo in product(*candidate_lists):
                binding = {}
                action_params = []
                for p, obj_name in zip(action.parameters, combo):
                    obj = self.up_objects[obj_name.lower()]
                    binding[p] = obj
                    action_params.append({"name": obj_name.upper(), "type": _up_type_name(p.type)})

                pre_atoms = []
                for pre in action.preconditions:
                    grounded_pre = pre.substitute(binding)
                    pre_atoms.extend(_collect_positive_atoms(grounded_pre, self.object_name_map))

                effect_atoms: list[tuple[str, bool]] = []
                for eff in action.effects:
                    if getattr(eff, "is_forall", lambda: False)():
                        forall_vars = eff.forall
                        if not forall_vars:
                            continue
                        var, = forall_vars
                        var_type = _up_type_name(var.type)
                        for obj_name in objects.get(var_type, []):
                            obj = self.up_objects[obj_name.lower()]
                            local_binding = dict(binding)
                            local_binding[var] = obj
                            if eff.condition is not None:
                                cond = eff.condition.substitute(local_binding)
                                if hasattr(cond, "is_false") and cond.is_false():
                                    continue
                                if hasattr(cond, "is_true") and cond.is_true():
                                    pass
                                else:
                                    cond_str = _infer_atom_string(cond, self.object_name_map)
                                    if cond_str is None or cond_str.startswith("(not "):
                                        continue
                                    if cond_str not in pre_atoms:
                                        continue
                            atom = _infer_atom_string(
                                eff.fluent.substitute(local_binding),
                                self.object_name_map,
                            )
                            if atom is None:
                                continue
                            val = eff.value
                            is_positive = hasattr(val, "is_true") and val.is_true()
                            effect_atoms.append((atom, is_positive))
                        continue

                    grounded_fluent = eff.fluent.substitute(binding)
                    atom = _infer_atom_string(grounded_fluent, self.object_name_map)
                    if atom is None:
                        continue
                    val = eff.value
                    is_positive = hasattr(val, "is_true") and val.is_true()
                    effect_atoms.append((atom, is_positive))

                pre_mask = [0] * n_atoms
                for atom_str in pre_atoms:
                    if atom_str in atom_str_to_idx:
                        pre_mask[atom_str_to_idx[atom_str]] = 1

                eff_delta = [0] * n_atoms
                for atom_str, is_positive in effect_atoms:
                    if atom_str in atom_str_to_idx:
                        eff_delta[atom_str_to_idx[atom_str]] = 1 if is_positive else -1

                semantics_list.append(ActionSemantics(
                    action_name=f"{action.name}({', '.join(combo)})",
                    parameters=action_params,
                    param_types=param_types,
                    precondition_mask=pre_mask,
                    effect_delta=eff_delta,
                    pre_atom_strs=pre_atoms,
                    effect_atom_strs=[a for a, _ in effect_atoms],
                ))

        return semantics_list

    def _find_dummy_problem(self, domain_path: str) -> str:
        """Find any problem file so PDDLReader can parse the domain."""
        path = Path(domain_path)
        root = path.parent
        candidates = []
        for pat in ("**/*.pddl", "**/*.pddl.*"):
            candidates.extend(root.glob(pat))
        for p in candidates:
            if p.name != path.name:
                return str(p)
        raise FileNotFoundError(f"Cannot find any PDDL problem under {root}")

    # ------------------------------------------------------------------
    # Gloss generation
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_gloss(
        name: str,
        arity: int,
        param_types: list[str],
        param_names: list[str],
    ) -> str:
        """Generate a natural language gloss for a predicate.

        Uses naming heuristics to create readable descriptions.
        Can be overridden with explicit glosses via PREDICATE_GLOSSES.
        """
        if name in PREDICATE_GLOSSES:
            template = PREDICATE_GLOSSES[name]
            if arity > 0:
                args = [f"{param_names[i]}" for i in range(arity)]
                try:
                    return template.format(*args)
                except (IndexError, KeyError):
                    return template
            return template

        # Heuristic gloss generation from name
        parts = re.split(r'[-_]', name)
        gloss_parts = []

        type_map = {
            "block": "block", "column": "column", "screw": "screw",
            "hole": "hole", "component": "component", "panel": "panel",
            "box": "box", "object": "object",
        }

        if arity == 0:
            return f"state flag: {name}"

        if arity == 1:
            return f"{name} holds for {{}} ({param_types[0]})"

        arg_desc = " and ".join(
            f"{{{i}}} ({param_types[i]})" for i in range(arity)
        )
        return f"{name}({arg_desc})"


# ======================================================================
# Predefined glosses for common predicates
# ======================================================================
PREDICATE_GLOSSES = {
    # Blocksworld
    "on": "block {0} is on top of block {1}",
    "clear": "block {0} has nothing on top",
    "inColumn": "block {0} is in column {1}",
    "rightOf": "column {0} is to the right of column {1}",
    "leftOf": "column {0} is to the left of column {1}",
    "onTable": "block {0} is on the table",
    "holding": "the hand is holding block {0}",
    "handEmpty": "the hand is empty",

    # TV Assembly
    "initial-state": "the assembly process is at the very beginning",
    "power-com-inspected": "the power component has been inspected",
    "power-com-placement-done": "the power component has been placed on the panel",
    "in-material-box": "component {0} is inside material box {1}",
    "comp-grasp-free": "the gripper is free, not holding any component",
    "screw-grasp-free": "the gripper is free, not holding any screw",
    "comp-in-hand": "component {0} is held in the gripper",
    "comp-at-panel-area": "component {0} is at panel {1} area",
    "comp-aligned": "component {0} is precisely aligned on panel {1}",
    "comp-on-panel": "component {0} is mounted on panel {1}",
    "screw-fetched": "a screw has been picked up",
    "screw-positioned": "a screw is positioned over its target hole",
    "screw-unused": "screw {0} has not been used yet",
    "screw-for-hole": "screw {0} is assigned to hole {1}",
    "hole-empty": "hole {0} is empty",
    "screw-aligned": "screw {0} is aligned with hole {1}",
    "screw-inserted": "screw {0} has been inserted into hole {1}",
    "screw-fastened": "screw {0} is fully fastened in hole {1}",
    "hole-done": "hole {0} is completed",
    "requires-predecessor": "hole {0} requires hole {1} to be done first",
}


# ======================================================================
# Data classes
# ======================================================================

class ObjectInfo:
    __slots__ = ("name", "type_name", "type_idx")

    def __init__(self, name: str, type_name: str, type_idx: int):
        self.name = name
        self.type_name = type_name
        self.type_idx = type_idx

    def __repr__(self):
        return f"ObjectInfo({self.name}: {self.type_name})"


class PredicateSchema:
    """Full PDDL predicate schema with action roles and gloss."""
    __slots__ = ("name", "arity", "param_types", "param_names", "action_roles", "gloss")

    def __init__(
        self,
        name: str,
        arity: int,
        param_types: list[str],
        param_names: list[str],
        action_roles: list[str],
        gloss: str,
    ):
        self.name = name
        self.arity = arity
        self.param_types = param_types
        self.param_names = param_names
        self.action_roles = action_roles
        self.gloss = gloss

    @property
    def schema_str(self) -> str:
        """PDDL-style schema string, e.g. 'on(?x:block, ?y:block)'."""
        params = ", ".join(
            f"{n}:{t}" for n, t in zip(self.param_names, self.param_types)
        )
        return f"{self.name}({params})" if params else f"{self.name}"

    def __repr__(self):
        return f"PredicateSchema({self.schema_str}, roles={self.action_roles})"


class GroundAtom:
    __slots__ = ("predicate", "arguments", "str_repr", "predicate_idx")

    def __init__(self, predicate: str, arguments: tuple, str_repr: str, predicate_idx: int):
        self.predicate = predicate
        self.arguments = arguments
        self.str_repr = str_repr
        self.predicate_idx = predicate_idx

    def __repr__(self):
        return self.str_repr


class ActionSemantics:
    """Action semantics for one grounded action instance.

    Provides both the legacy ``effect_delta`` (add=+1, del=-1) and the
    AE-PaQ style explicit masks:
        - ``add_mask``      : 1 where action adds the atom
        - ``del_mask``      : 1 where action deletes the atom
        - ``frame_mask``    : 1 where action does NOT change the atom
    """
    __slots__ = (
        "action_name", "parameters", "param_types",
        "precondition_mask", "effect_delta",
        "add_mask", "del_mask", "frame_mask",
        "pre_atom_strs", "effect_atom_strs",
    )

    def __init__(
        self,
        action_name: str,
        parameters: list[dict],
        param_types: list[str],
        precondition_mask: list[int],
        effect_delta: list[int],
        pre_atom_strs: list[str],
        effect_atom_strs: list[str],
    ):
        self.action_name = action_name
        self.parameters = parameters
        self.param_types = param_types
        self.precondition_mask = precondition_mask
        self.effect_delta = effect_delta
        self.pre_atom_strs = pre_atom_strs
        self.effect_atom_strs = effect_atom_strs

        # Derive explicit masks from effect_delta
        self.add_mask = [1 if d > 0 else 0 for d in effect_delta]
        self.del_mask = [1 if d < 0 else 0 for d in effect_delta]
        self.frame_mask = [
            0 if (d > 0 or d < 0) else 1 for d in effect_delta
        ]

    def __repr__(self):
        return (
            f"ActionSemantics({self.action_name}, "
            f"pre={sum(1 for x in self.precondition_mask if x)}, "
            f"eff_add={sum(1 for x in self.effect_delta if x > 0)}, "
            f"eff_del={sum(1 for x in self.effect_delta if x < 0)}, "
            f"frame={sum(self.frame_mask)})"
        )


class DomainInfo:
    """Complete compiled domain information ready for model consumption."""

    def __init__(
        self,
        domain_name: str,
        types: list[str],
        type_to_idx: dict[str, int],
        objects: list[ObjectInfo],
        obj_name_to_idx: dict[str, int],
        predicate_schemas: list[PredicateSchema],
        canonical_atoms: list[GroundAtom],
        action_semantics: list[ActionSemantics],
        obj_type_ids: list[int],
        static_predicates: set[str],
        n_canonical: int,
    ):
        self.domain_name = domain_name
        self.types = types
        self.type_to_idx = type_to_idx
        self.objects = objects
        self.obj_name_to_idx = obj_name_to_idx
        self.predicate_schemas = predicate_schemas
        self.canonical_atoms = canonical_atoms
        self.action_semantics = action_semantics
        self.obj_type_ids = obj_type_ids
        self.static_predicates = static_predicates
        self.n_canonical = n_canonical

    @property
    def canonical_atom_strings(self) -> list[str]:
        return [a.str_repr for a in self.canonical_atoms]

    @property
    def predicate_names(self) -> list[str]:
        return [s.name for s in self.predicate_schemas]

    @property
    def predicate_arities(self) -> dict[str, int]:
        return {s.name: s.arity for s in self.predicate_schemas}

    @property
    def predicate_param_types(self) -> dict[str, list[str]]:
        return {s.name: s.param_types for s in self.predicate_schemas}

    @property
    def predicate_defs(self) -> list[dict]:
        """Format compatible with PredicateScoringHead."""
        return [
            {"name": s.name, "arity": s.arity, "param_types": s.param_types}
            for s in self.predicate_schemas
        ]

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_predicates(self) -> int:
        return len(self.predicate_schemas)

    @property
    def n_types(self) -> int:
        return len(self.types)

    def get_action_masks_tensor(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Return all action masks as stacked tensors for GPU computation.

        Returns:
            dict with keys:
                'precondition_mask': (N_actions, N_canonical)
                'add_mask':          (N_actions, N_canonical)
                'del_mask':          (N_actions, N_canonical)
                'frame_mask':        (N_actions, N_canonical)
                'effect_delta':      (N_actions, N_canonical)
        """
        import torch
        n = self.n_canonical
        sems = self.action_semantics
        if not sems:
            return {}
        pre = torch.tensor([s.precondition_mask for s in sems], dtype=torch.float32, device=device)
        add = torch.tensor([s.add_mask for s in sems], dtype=torch.float32, device=device)
        dl  = torch.tensor([s.del_mask for s in sems], dtype=torch.float32, device=device)
        fr  = torch.tensor([s.frame_mask for s in sems], dtype=torch.float32, device=device)
        ed  = torch.tensor([s.effect_delta for s in sems], dtype=torch.float32, device=device)
        return {
            "precondition_mask": pre,
            "add_mask": add,
            "del_mask": dl,
            "frame_mask": fr,
            "effect_delta": ed,
        }

    def get_action_name_to_idx(self) -> dict[str, int]:
        """Map grounded action name -> index into action_semantics list."""
        return {s.action_name: i for i, s in enumerate(self.action_semantics)}

    def summary(self) -> dict:
        return {
            "domain": self.domain_name,
            "types": self.types,
            "n_objects": self.n_objects,
            "n_predicates": self.n_predicates,
            "n_canonical": self.n_canonical,
            "predicates": [
                {"schema": s.schema_str, "roles": s.action_roles, "gloss": s.gloss}
                for s in self.predicate_schemas
            ],
            "actions": [repr(a) for a in self.action_semantics],
        }

    def __repr__(self):
        return (
            f"DomainInfo({self.domain_name}, "
            f"types={self.types}, "
            f"objs={self.n_objects}, "
            f"preds={self.n_predicates}, "
            f"atoms={self.n_canonical})"
        )
