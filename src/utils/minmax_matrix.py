"""
This Matrix class is used primarily for getting the maximum
and minimum field within it. It is optimized in the sense that when we
delete a row or column, those rows or columns are only soft deleted
"""


class MinMaxMatrix:
    def __init__(self, matrix: list[list[float]]) -> None:
        self.matrix = matrix
        self.active_rows = set(range(len(self.matrix)))
        self.active_columns = \
            set() if len(self.active_rows) == 0 \
                else set(range(len(self.matrix[0])))

    def soft_delete_row(self, original_index: int) -> None:
        self.active_rows.remove(original_index)

    def soft_delete_column(self, original_index: int) -> None:
        self.active_columns.remove(original_index)

    def get_active_rows(self) -> set[int]:
        return self.active_rows

    def get_active_columns(self) -> set[int]:
        return self.active_columns

    def get_min_field(self) -> tuple[float, int, int]:
        min_row = min_column = -1
        min_value = float('inf')
        for row in self.active_rows:
            for col in self.active_columns:
                current_value = self.matrix[row][col]
                if current_value < min_value:
                    min_row, min_column, min_value = row, col, current_value
        return min_row, min_column, min_value

    def get_max_field(self) -> tuple[float, int, int]:
        max_row = max_column = -1
        max_value = float('-inf')
        for row in self.active_rows:
            for col in self.active_columns:
                current_value = self.matrix[row][col]
                if current_value > max_value:
                    max_row, max_column, max_value = row, col, current_value
        return max_row, max_column, max_value
