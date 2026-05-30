"""
PDDL Domain Parser
===================
Parses a PDDL domain file and extracts types, predicates (with parameter signatures),
and actions (with preconditions and effects). Provides grounding of predicates given
a set of typed objects.
"""

import re
import json
from itertools import product
from typing import Any


class PDDLParser:
    """Parse a PDDL domain file into a structured representation."""

    def __init__(self, domain_path: str):
        with open(domain_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        self.domain_name = ""
        self.requirements: list[str] = []
        self.types: list[str] = []
        self.predicates: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self._parse()

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_comments(text: str) -> str:
        """Remove ;; comments."""
        return re.sub(r";[^\n]*", "", text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Parentheses-aware tokenizer."""
        tokens: list[str] = []
        buf: list[str] = []
        for ch in text:
            if ch in ("(", ")"):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            elif ch.isspace():
                if buf:
                    tokens.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))
        return tokens

    @staticmethod
    def _sexpr(tokens: list[str], start: int = 0) -> tuple[list, int]:
        """Parse an S-expression from *tokens* starting at *start*.

        Returns (parsed_list, next_index).
        """
        if tokens[start] != "(":
            raise SyntaxError(f"Expected '(' at position {start}, got '{tokens[start]}'")
        result: list = []
        i = start + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == ")":
                return result, i + 1
            elif tok == "(":
                sub, i = PDDLParser._sexpr(tokens, i)
                result.append(sub)
            else:
                result.append(tok)
                i += 1
        raise SyntaxError("Unmatched '('")

    # ------------------------------------------------------------------
    # High-level parsing
    # ------------------------------------------------------------------
    def _parse(self):
        clean = self._strip_comments(self.raw_text)
        tokens = self._tokenize(clean)
        tree, _ = self._sexpr(tokens, 0)

        # tree[0] should be "define"
        assert tree[0] == "define", f"Expected 'define', got '{tree[0]}'"

        for elem in tree[1:]:
            tag = elem[0] if isinstance(elem, list) and elem else None
            if tag == "domain":
                self.domain_name = elem[1]
            elif tag == ":requirements":
                self.requirements = elem[1:]
            elif tag == ":types":
                self.types = self._parse_types(elem[1:])
            elif tag == ":predicates":
                self.predicates = self._parse_predicates(elem[1:])
            elif tag == ":action":
                self.actions.append(self._parse_action(elem))

    # ------------------------------------------------------------------
    # Types
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_types(tokens: list) -> list[str]:
        """Parse ``component panel box screw hole`` style type lists.

        Handles simple ``t1 t2 ... - parent`` syntax, returning a flat list
        of leaf type names (ignores hierarchy for grounding purposes).
        """
        types: list[str] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-":
                i += 2  # skip parent type
            else:
                types.append(tokens[i])
                i += 1
        return types

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_predicates(elems: list) -> list[dict]:
        preds = []
        for elem in elems:
            name = elem[0]
            params = []
            i = 1
            while i < len(elem):
                if elem[i] == "-":
                    # previous token(s) are param names for the type that follows
                    i += 1  # skip to type
                    ptype = elem[i]
                    # assign type to pending param names
                    for p in params:
                        if p["type"] is None:
                            p["type"] = ptype
                    i += 1
                else:
                    pname = elem[i]
                    params.append({"name": pname, "type": None})
                    i += 1
            preds.append({"name": name, "params": params})
        return preds

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _parse_action(self, elem: list) -> dict:
        action = {"name": elem[1]}
        i = 2
        while i < len(elem):
            tag = elem[i]
            if tag == ":parameters":
                action["parameters"] = self._parse_params(elem[i + 1])
                i += 2
            elif tag == ":precondition":
                action["precondition"] = elem[i + 1]
                i += 2
            elif tag == ":effect":
                action["effect"] = elem[i + 1]
                i += 2
            else:
                i += 1
        return action

    @staticmethod
    def _parse_params(elem: list) -> list[dict]:
        """Parse ``(?c - component ?b - box)``."""
        params: list[dict] = []
        i = 0
        while i < len(elem):
            if elem[i] == "-":
                i += 1
                continue
            if elem[i].startswith("?"):
                pname = elem[i]
                # look ahead for type
                ptype = "object"
                if i + 2 < len(elem) and elem[i + 1] == "-":
                    ptype = elem[i + 2]
                    i += 3
                else:
                    i += 1
                params.append({"name": pname, "type": ptype})
            else:
                i += 1
        return params

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    def get_all_ground_predicates(self, objects: dict[str, list[str]]) -> list[str]:
        """Return every possible ground predicate string for the given typed objects.

        Parameters
        ----------
        objects : dict mapping type_name -> list of object names
            e.g. {"component": ["power_com"], "panel": ["TV_panel"], ...}

        Returns
        -------
        list of strings like ``"(in-material-box power_com material_box)"``
        """
        grounded: list[str] = []
        for pred in self.predicates:
            if not pred["params"]:
                # nullary predicate
                grounded.append(f"({pred['name']})")
                continue
            # collect candidate lists for each param
            param_choices: list[list[str]] = []
            for p in pred["params"]:
                ptype = p["type"] or "object"
                candidates = objects.get(ptype, [])
                if not candidates:
                    param_choices = []
                    break
                param_choices.append(candidates)
            if not param_choices:
                continue
            for combo in product(*param_choices):
                args = " ".join(combo)
                grounded.append(f"({pred['name']} {args})")
        return grounded

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def domain_summary(self) -> dict:
        return {
            "domain_name": self.domain_name,
            "requirements": self.requirements,
            "types": self.types,
            "predicates": self.predicates,
            "actions": [
                {
                    "name": a["name"],
                    "parameters": a.get("parameters", []),
                }
                for a in self.actions
            ],
        }

    def __repr__(self):
        return (
            f"PDDLParser(domain={self.domain_name}, "
            f"types={len(self.types)}, "
            f"predicates={len(self.predicates)}, "
            f"actions={len(self.actions)})"
        )


# ======================================================================
# Problem file parser (lightweight -- extracts objects and init state)
# ======================================================================

class PDDLProblemParser:
    """Parse a PDDL problem file to extract objects and the initial state."""

    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        self.problem_name = ""
        self.domain_ref = ""
        self.objects: dict[str, list[str]] = {}
        self.init_state: list[str] = []
        self.goal_state: list[str] = []
        self._parse()

    def _parse(self):
        clean = PDDLParser._strip_comments(self.raw_text)
        tokens = PDDLParser._tokenize(clean)
        tree, _ = PDDLParser._sexpr(tokens, 0)  # borrow static methods

        for elem in tree[1:]:
            tag = elem[0] if isinstance(elem, list) and elem else None
            if tag == "problem":
                self.problem_name = elem[1]
            elif tag == ":domain":
                self.domain_ref = elem[1]
            elif tag == ":objects":
                self._parse_objects(elem[1:])
            elif tag == ":init":
                self.init_state = self._parse_atoms(elem[1:])
            elif tag == ":goal":
                self.goal_state = self._parse_atoms_from_expr(elem[1])

    def _parse_objects(self, tokens: list):
        i = 0
        pending: list[str] = []
        while i < len(tokens):
            if tokens[i] == "-":
                i += 1
                ptype = tokens[i]
                self.objects.setdefault(ptype, []).extend(pending)
                pending = []
                i += 1
            else:
                pending.append(tokens[i])
                i += 1

    @staticmethod
    def _atom_to_str(atom) -> str:
        if isinstance(atom, str):
            return f"({atom})"
        parts = [str(x) for x in atom]
        return f"({' '.join(parts)})"

    def _parse_atoms(self, elems: list) -> list[str]:
        return [self._atom_to_str(e) for e in elems]

    def _parse_atoms_from_expr(self, expr) -> list[str]:
        """Recursively extract positive atoms from an expression (e.g. ``(and ...)``)."""
        if isinstance(expr, str):
            return [f"({expr})"]
        tag = expr[0] if expr else None
        if tag == "and":
            atoms: list[str] = []
            for sub in expr[1:]:
                atoms.extend(self._parse_atoms_from_expr(sub))
            return atoms
        if tag == "not":
            return []  # skip negated atoms for goal
        return [self._atom_to_str(expr)]


# ======================================================================
# Quick self-test
# ======================================================================

if __name__ == "__main__":
    domain = PDDLParser("/home/pc/PDDL/solver/domain.pddl")
    print(domain)
    print(json.dumps(domain.domain_summary(), indent=2, ensure_ascii=False))

    problem = PDDLProblemParser("/home/pc/PDDL/solver/p_real.pddl")
    print(f"\nObjects: {json.dumps(problem.objects, indent=2)}")
    print(f"\nInit state ({len(problem.init_state)} atoms):")
    for a in problem.init_state:
        print(f"  {a}")

    grounded = domain.get_all_ground_predicates(problem.objects)
    print(f"\nTotal grounded predicates: {len(grounded)}")
