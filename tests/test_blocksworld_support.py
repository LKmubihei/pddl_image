from pathlib import Path

import torch

from paq.blocksworld_support import BlocksworldSupportSketch
from paq.domain_compiler import PDDLDomainCompiler


ROOT = Path(__file__).resolve().parents[1]
STATIC_PREDS = {"rightof", "leftof"}


def _sketch():
    compiler = PDDLDomainCompiler(str(ROOT / "data/planning/blocksworld/domain.pddl"))
    domain_info = compiler.compile(
        objects={"block": ["Y", "P", "R", "O"], "column": ["C1", "C2", "C3", "C4"]},
        static_predicates=STATIC_PREDS,
    )
    return BlocksworldSupportSketch.from_domain_info(domain_info)


def test_derive_atoms_and_recover_support_targets():
    sketch = _sketch()
    assignment = {"Y": "C1", "P": "Y", "R": "C3", "O": "P"}

    atoms = sketch.derive_atoms(assignment)
    assert "(on P Y)" in atoms
    assert "(on O P)" in atoms
    assert "(inColumn O C1)" in atoms
    assert "(inColumn R C3)" in atoms
    assert "(clear O)" in atoms
    assert "(clear R)" in atoms
    assert "(clear Y)" not in atoms

    labels = torch.tensor(sketch.assignment_to_vector(assignment)).float()
    targets = sketch.labels_to_support_targets(labels)
    recovered = {
        b: sketch.support_candidates[b][int(targets[i].item())]
        for i, b in enumerate(sketch.blocks)
    }
    assert recovered == assignment


def test_assignment_constraints_reject_cycles_and_fan_in():
    sketch = _sketch()

    assert not sketch.is_valid_assignment(
        {"Y": "P", "P": "Y", "R": "C3", "O": "C4"}
    )
    assert not sketch.is_valid_assignment(
        {"Y": "R", "P": "R", "R": "C3", "O": "C4"}
    )


def test_decoder_avoids_illegal_high_local_cycle():
    sketch = _sketch()
    scores = torch.zeros(sketch.n_blocks, sketch.n_candidates)

    def set_score(block, support, value):
        bi = sketch.blocks.index(block)
        ci = sketch.support_candidates[block].index(support)
        scores[bi, ci] = value

    # Locally attractive but globally illegal 2-cycle.
    set_score("Y", "P", 10.0)
    set_score("P", "Y", 10.0)

    # Best legal alternative.
    set_score("Y", "C1", 9.0)
    set_score("P", "Y", 10.0)
    set_score("R", "C3", 5.0)
    set_score("O", "P", 4.0)

    decoded = sketch.decode(scores)
    assert decoded.assignment == {"Y": "C1", "P": "Y", "R": "C3", "O": "P"}
    assert "(on P Y)" in decoded.atoms
    assert "(on Y P)" not in decoded.atoms

