import torch
import torch.optim as optim
import numpy as np
import sys

from Modeling.HRM import HRMACTInner, HRMACTModelConfig
from tranning import TrainingBatch, Difficulty, step
from Sudoku import generate_sudoku, sudoku_board_string


def train():
    torch.manual_seed(42)

    # model setup
    model_config = HRMACTModelConfig(
        seq_len=9 * 9,
        vocab_size=10,
        high_level_cycles=2,
        low_level_cycles=2,
        transformers=HRMACTModelConfig.TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_heads=4,
            expansion=4
        ),
        act=HRMACTModelConfig.ACTConfig(
            halt_max_steps=16,
            halt_exploration_probability=0.1
        ),
    )

    model = HRMACTInner(model_config)
    model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    batch = TrainingBatch(initial_hidden_states=model.initial_hidden_states(), size=512)

    step_idx = 0
    steps_since_graduation = 0
    accuracy_history = [0.0] * 300

    while True:
        step_idx += 1
        steps_since_graduation += 1
        print(f"Step {step_idx}")

        (_, output_acc), _ = step(model=model, optimizer=optimizer, batch=batch)

        # checkpoint save
        if step_idx == 1 or step_idx % 250 == 0:
            ckpt = f"checkpoint-{step_idx}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint {ckpt}")

        # rolling accuracy
        accuracy_history.pop(0)
        accuracy_history.append(output_acc)
        avg_rolling_accuracy = sum(accuracy_history) / len(accuracy_history)

        if avg_rolling_accuracy >= 0.85 and steps_since_graduation >= 300:
            steps_since_graduation = 0
            batch.graduate()

        print()


def infer(checkpoint: str, difficulty: Difficulty):
    torch.manual_seed(42)

    # load weights
    model_config = HRMACTModelConfig(
            seq_len=9 * 9,
            vocab_size=10,
            high_level_cycles=2,
            low_level_cycles=2,
            transformers=HRMACTModelConfig.TransformerConfig(
                num_layers=4,
                hidden_size=256,
                num_heads=4,
                expansion=4
            ),
            act=HRMACTModelConfig.ACTConfig(
                halt_max_steps=16,
                halt_exploration_probability=0.1
            ),
        )
    model = HRMACTInner(model_config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print("Loaded model!")

    # sudoku puzzle
    puzzle, solution = generate_sudoku(difficulty)
    print("Puzzle:\n" + sudoku_board_string(puzzle))
    print("Solution:\n" + sudoku_board_string(solution))

    puzzle_in = torch.tensor(np.array(puzzle).flatten(), dtype=torch.int32).unsqueeze(0)
    hidden_states = model.initial_hidden_states()
    hidden_states.high_level = hidden_states.high_level.unsqueeze(0).unsqueeze(0)
    hidden_states.low_level  = hidden_states.low_level.unsqueeze(0).unsqueeze(0)

    for segment in range(1, model.config.act.halt_max_steps + 1):
        print(f"\nSegment {segment}")

        output = model(hidden_states=hidden_states, inputs=puzzle_in)
        hidden_states = output.hidden_states

        predictions = output.output[0].argmax(dim=1)

        accurate_squares = 0
        predicted_squares = 0
        predicted_flat_board = []
        for puzz_val, sol_val, pred_val in zip(np.array(puzzle).flatten(),
                                               np.array(solution).flatten(),
                                               predictions.tolist()):
            if puzz_val != 0:
                predicted_flat_board.append(puzz_val)
            else:
                if pred_val == sol_val:
                    accurate_squares += 1
                predicted_squares += 1
                predicted_flat_board.append(pred_val)

        predicted_board = [predicted_flat_board[i:i+9] for i in range(0, 81, 9)]
        print(f"Predicted solution ({accurate_squares} / {predicted_squares}):\n"
              + sudoku_board_string(predicted_board))

        q_halt = torch.sigmoid(output.qACTHalt[0]).item()
        q_continue = torch.sigmoid(output.qACTContinue[0]).item()
        print(f"Q (halt - continue): {q_halt:.4f} - {q_continue:.4f}")

        if q_halt > q_continue:
            print("Halting.")
            break


def main():
    args = sys.argv
    if len(args) < 2:
        print(f"Usage: {args[0]} train | infer <checkpoint-path> <difficulty>")
        return

    mode = args[1]

    if mode == "train":
        train()
    elif mode == "infer":
        if len(args) < 4:
            print(f"Usage: {args[0]} infer <checkpoint-path> <difficulty>")
            return

        checkpoint = args[2]
        diff_str = args[3]

        difficulty_map = {
            "very-easy": Difficulty.VERY_EASY,
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
            "extreme": Difficulty.EXTREME,
        }
        if diff_str not in difficulty_map:
            raise ValueError(f"Unknown difficulty: {diff_str}. "
                             f"Expected one of {', '.join(difficulty_map.keys())}.")

        infer(checkpoint=checkpoint, difficulty=difficulty_map[diff_str])
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'train' or 'infer'.")


if __name__ == "__main__":
    main()
