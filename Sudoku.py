import random
from enum import Enum
from typing import List, Optional, Tuple


class Difficulty(Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXTREME = 5


def bit(value: int) -> int:
    return 1 << (value - 1)


def box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


def build_masks(grid: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9

    for r in range(9):
        row_mask = 0
        for c in range(9):
            v = grid[r][c]
            if v != 0:
                b = bit(v)
                row_mask |= b
                cols[c] |= b
                boxes[box_index(r, c)] |= b
        rows[r] = row_mask
    return rows, cols, boxes


ALL_DIGITS = list(range(1, 10))


def fill_grid(grid: List[List[int]]) -> bool:
    rows, cols, boxes = build_masks(grid)
    return fill_grid_rec(grid, rows, cols, boxes)


def fill_grid_rec(grid, rows, cols, boxes) -> bool:
    empty = first_empty_cell(grid)
    if empty is None:
        return True
    row, col = empty

    numbers = ALL_DIGITS[:]
    random.shuffle(numbers)

    b_idx = box_index(row, col)
    used = rows[row] | cols[col] | boxes[b_idx]

    for num in numbers:
        b = bit(num)
        if used & b:
            continue

        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[b_idx] |= b

        if fill_grid_rec(grid, rows, cols, boxes):
            return True

        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[b_idx] &= ~b
    return False


def solve(grid: List[List[int]], count: List[int], stop_after: int) -> bool:
    rows, cols, boxes = build_masks(grid)
    return solve_rec(grid, count, stop_after, rows, cols, boxes)


def solve_rec(grid, solutions, limit, rows, cols, boxes) -> bool:
    if solutions[0] >= limit:
        return True
    empty = first_empty_cell(grid)
    if empty is None:
        solutions[0] += 1
        return solutions[0] >= limit
    row, col = empty

    b_idx = box_index(row, col)
    used = rows[row] | cols[col] | boxes[b_idx]

    for num in range(1, 10):
        b = bit(num)
        if used & b:
            continue

        grid[row][col] = num
        rows[row] |= b
        cols[col] |= b
        boxes[b_idx] |= b

        if solve_rec(grid, solutions, limit, rows, cols, boxes):
            return True

        grid[row][col] = 0
        rows[row] &= ~b
        cols[col] &= ~b
        boxes[b_idx] &= ~b
    return False


def first_empty_cell(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def generate_sudoku(difficulty: Difficulty) -> Tuple[List[List[int]], List[List[int]]]:
    board = [[0] * 9 for _ in range(9)]
    fill_grid(board)
    solution = [row[:] for row in board]

    target_clues = {
        Difficulty.VERY_EASY: range(46, 51),
        Difficulty.EASY: range(40, 46),
        Difficulty.MEDIUM: range(32, 40),
        Difficulty.HARD: range(28, 32),
        Difficulty.EXTREME: range(17, 28),
    }[difficulty]

    puzzle = [row[:] for row in board]
    cells = list(range(81))
    random.shuffle(cells)

    cursor = 0
    clues = 81

    while cursor < len(cells) and clues > max(target_clues):
        idx = cells[cursor]
        cursor += 1
        r, c = divmod(idx, 9)
        backup = puzzle[r][c]
        puzzle[r][c] = 0

        test = [row[:] for row in puzzle]
        solution_counter = [0]
        solve(test, solution_counter, 2)
        if solution_counter[0] != 1:
            puzzle[r][c] = backup
        else:
            clues -= 1

    if clues > min(target_clues):
        for j in range(cursor, len(cells)):
            if clues <= min(target_clues):
                break
            idx = cells[j]
            r, c = divmod(idx, 9)
            backup = puzzle[r][c]
            puzzle[r][c] = 0

            test = [row[:] for row in puzzle]
            solution_counter = [0]
            solve(test, solution_counter, 2)
            if solution_counter[0] != 1:
                puzzle[r][c] = backup
            else:
                clues -= 1

    return puzzle, solution


def sudoku_board_string(board: List[List[int]]) -> str:
    horizontal_line = "+-------+-------+-------+"
    result = horizontal_line + "\n"

    for row_idx, row in enumerate(board):
        line = "|"
        for col_idx, cell in enumerate(row):
            display_value = "." if cell == 0 else str(cell)
            line += f" {display_value}"
            if (col_idx + 1) % 3 == 0:
                line += " |"
        result += line + "\n"
        if (row_idx + 1) % 3 == 0:
            result += horizontal_line + "\n"

    return result.strip()


if __name__ == "__main__":
    puzzle, solution = generate_sudoku(Difficulty.MEDIUM)
    print("Puzzle:")
    print(sudoku_board_string(puzzle))
    print("\nSolution:")
    print(sudoku_board_string(solution))
