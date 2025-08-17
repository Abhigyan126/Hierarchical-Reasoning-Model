import torch
import torch.nn.functional as F
import random

from Modeling.HRM import HRMACTInner
from Sudoku import generate_sudoku, Difficulty


# -------------------------
# Sudoku Loss
# -------------------------
def sudoku_loss(model, hidden_states, board_inputs, board_targets, segments, key=None):
    output = model(hidden_states=hidden_states, inputs=board_inputs)

    output_logits = output.output
    output_loss = F.cross_entropy(
        output_logits.transpose(1, 2),  # (B, V, T) for CE
        board_targets,
        reduction="none",
    )

    output_loss_mask = (board_inputs == 0).to(output_loss.dtype)
    output_loss = output_loss * output_loss_mask

    output_accuracy = (
        ((output.output.argmax(dim=2) == board_targets) | (board_inputs != 0))
        .amin(dim=1)
        .to(torch.int32)
    )
    qact_halt_target = output_accuracy

    next_segments = segments + 1
    is_last_segment = next_segments > model.config.act.haltMaxSteps
    is_halted = is_last_segment | (output.qACTHalt > output.qACTContinue)

    # Exploration
    halt_exploration = (torch.rand_like(output.qACTHalt) <
                        model.config.act.haltExplorationProbability)
    min_halt_segments = (
        torch.randint(
            low=2,
            high=model.config.act.haltMaxSteps + 1,
            size=segments.shape,
        ) * halt_exploration.int()
    )
    is_halted = is_halted & (next_segments > min_halt_segments)

    # Next step (stop_gradient equivalent = detach)
    next_output = model(hidden_states=output.hiddenStates, inputs=board_inputs)
    next_qact_halt = next_output.qACTHalt.detach()
    next_qact_continue = next_output.qACTContinue.detach()

    qact_continue_target = torch.sigmoid(
        torch.where(
            is_last_segment,
            next_qact_halt,
            torch.maximum(next_qact_halt, next_qact_continue),
        )
    )

    qact_loss = (
        F.binary_cross_entropy_with_logits(
            output.qACTHalt, qact_halt_target.float(), reduction="none"
        )
        + F.binary_cross_entropy_with_logits(
            output.qACTContinue, qact_continue_target, reduction="none"
        )
    ) / 2

    avg_output_loss = output_loss.sum() / output_loss_mask.sum()
    avg_qact_loss = qact_loss.mean()

    avg_output_full_accuracy = (
        ((output.output.argmax(dim=2) == board_targets) | (board_inputs != 0))
        .float()
        .mean()
    )
    avg_qact_halt_accuracy = ((output.qACTHalt >= 0).int() == output_accuracy).float().mean()

    return [
        avg_output_loss + avg_qact_loss,
        avg_output_loss,
        avg_qact_loss,
        is_halted,
        avg_output_full_accuracy,
        avg_qact_halt_accuracy,
        output.hiddenStates.highLevel,
        output.hiddenStates.lowLevel,
    ]


# -------------------------
# Training Batch
# -------------------------
class TrainingBatch:
    DIFFICULTIES = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXTREME]

    CURRICULUM_DIFFICULTY_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],  # stage 0: only easy
        [0.7, 0.3, 0.0, 0.0],  # stage 1: mostly easy, some medium
        [0.5, 0.4, 0.1, 0.0],  # stage 2: mix of easy, medium, some hard
        [0.3, 0.3, 0.3, 0.1],  # stage 3: mix of all difficulties
        [0.1, 0.3, 0.4, 0.2],  # stage 4: more hard and extreme
    ]

    def __init__(self, initial_hidden_states, size: int):
        self.initial_hidden_states = initial_hidden_states
        self.hidden_states = HRMACTInner.HiddenStates(
                high_level=self.initial_hidden_states.high_level.unsqueeze(0).unsqueeze(0).expand(size, -1, -1).clone(),
                low_level=self.initial_hidden_states.low_level.unsqueeze(0).unsqueeze(0).expand(size, -1, -1).clone(),
        )
        self.board_inputs = torch.zeros(size, 81, dtype=torch.int32)
        self.board_targets = torch.zeros(size, 81, dtype=torch.long)
        self.segments = torch.zeros(size, dtype=torch.int32)

        self.curriculum_level = 0
        self.total_puzzles = 0

        for i in range(size):
            self.replace(i)

    def inner_state(self):
        return (
            self.hidden_states.inner_state()
            + [self.board_inputs, self.board_targets, self.segments]
        )

    def sample_difficulty(self):
        probabilities = self.CURRICULUM_DIFFICULTY_PROBAS[self.curriculum_level]
        rand = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if rand < cumulative:
                return self.DIFFICULTIES[idx]
        raise RuntimeError("Invalid probability distribution.")

    def replace(self, idx: int):
        self.hidden_states.high_level[idx] = self.initial_hidden_states.high_level
        self.hidden_states.low_level[idx]  = self.initial_hidden_states.low_level
        self.segments[idx] = 0

        puzzle, solution = generate_sudoku(self.sample_difficulty())
        self.board_inputs[idx] = torch.tensor([x for row in puzzle for x in row],
                                              dtype=torch.int32)
        self.board_targets[idx] = torch.tensor([x for row in solution for x in row],
                                               dtype=torch.long)

        self.total_puzzles += 1

    def graduate(self):
        next_level = self.curriculum_level + 1
        if next_level >= len(self.CURRICULUM_DIFFICULTY_PROBAS):
            print("Reached highest curriculum level, cannot graduate.")
            return
        self.curriculum_level = next_level
        print(f"Graduated to level {self.curriculum_level}.")


# -------------------------
# Training Step
# -------------------------
def step(model, optimizer, batch: TrainingBatch, key=None):
    def closure():
        return sudoku_loss(
            model,
            batch.hidden_states,
            batch.board_inputs,
            batch.board_targets,
            batch.segments,
            key=key,
        )

    # valueAndGrad equivalent
    losses = closure()
    (loss,) = losses[0:1]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output_loss, qact_loss, output_acc, qact_acc = (
        losses[1].item(),
        losses[2].item(),
        losses[4].item(),
        losses[5].item(),
    )

    print(
        f"Output [{output_loss:.4f} {output_acc:.4f}] | "
        f"Q-ACT [{qact_loss:.4f} {qact_acc:.4f}] | "
        f"Puzzles [{batch.total_puzzles}] | "
        f"Curriculum Level [{batch.curriculum_level}]"
    )

    next_hl, next_ll = losses[6], losses[7]
    batch.hidden_states.high_level = next_hl
    batch.hidden_states.low_level = next_ll
    batch.segments += 1

    is_halted = losses[3].cpu().numpy().astype(bool)
    for idx, halted in enumerate(is_halted):
        if halted:
            batch.replace(idx)

    return (output_loss, output_acc), (qact_loss, qact_acc)
