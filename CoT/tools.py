from langchain_core.tools import tool
from typing import Annotated, Sequence

class ToolExecutor:
    def __init__(self, tools):
        """
        Initialize ToolExecutor with available tools.
        :param tools: Dictionary of tool names and corresponding tool functions.
        """
        self.tools = tools

    def execute(self, tool_name, *args, **kwargs):
        """
        Execute the specified tool with given arguments.
        :param tool_name: The name of the tool to execute.
        :param args: Positional arguments for the tool.
        :param kwargs: Keyword arguments for the tool.
        :return: The result of the tool execution.
        """
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error executing tool '{tool_name}': {e}")
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")


class Sorter:
    @tool
    def sort(
        data: Annotated[Sequence[dict], "List of dictionaries representing rows"],
        column: Annotated[str, "The column by which to sort"],
        ascending: Annotated[bool, "Whether to sort in ascending order"] = True
    ):
        """
        Sort the given data by the specified column.
        :param data: List of dictionaries representing rows.
        :param column: The column by which to sort.
        :param ascending: Whether to sort in ascending order.
        :return: Sorted data.
        """
        return sorted(data, key=lambda x: x[column], reverse=not ascending)


class Filter:
    @tool
    def filter(
        data: Annotated[Sequence[dict], "List of dictionaries representing rows"],
        column: Annotated[str, "The column to filter by"],
        value: Annotated[object, "The value to filter for"]
    ):
        """
        Filter rows based on the value of a specific column.
        :param data: List of dictionaries representing rows.
        :param column: The column to filter by.
        :param value: The value to filter for.
        :return: Filtered data.
        """
        return [row for row in data if row[column] == value]


class Calculator:
    @tool
    def calculate(
        expression: Annotated[str, "A string representing the arithmetic expression"]
    ):
        """
        Evaluate a simple arithmetic expression.
        :param expression: A string representing the arithmetic expression.
        :return: The result of the evaluation.
        """
        try:
            return eval(expression)
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}") from e


class Aggregator:
    @tool
    def sum_column(
        data: Annotated[Sequence[dict], "List of dictionaries representing rows"],
        column: Annotated[str, "The column for which to calculate the sum"]
    ):
        """
        Calculate the sum of a specific column in the data.
        :param data: List of dictionaries representing rows.
        :param column: The column for which to calculate the sum.
        :return: The sum of the values in the column.
        """
        return sum(row[column] for row in data if isinstance(row[column], (int, float)))

    @tool
    def count_rows(
        data: Annotated[Sequence[dict], "List of dictionaries representing rows"]
    ):
        """
        Count the number of rows in the data.
        :param data: List of dictionaries representing rows.
        :return: The number of rows.
        """
        return len(data)

    @tool
    def find_mode(
        data: Annotated[Sequence[dict], "List of dictionaries representing rows"],
        column: Annotated[str, "The column for which to find the mode"]
    ):
        """
        Find the most common value in a specific column.
        :param data: List of dictionaries representing rows.
        :param column: The column for which to find the mode.
        :return: The most common value.
        """
        from collections import Counter
        values = [row[column] for row in data]
        most_common = Counter(values).most_common(1)
        return most_common[0][0] if most_common else None
