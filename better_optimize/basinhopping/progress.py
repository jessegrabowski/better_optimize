import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

from better_optimize.constants import CONSOLE_WIDTH
from better_optimize.utilities import ToggleableProgress


def initialize_progress_bar(progressbar, use_jac=True, use_hess=False):
    description = "Process"
    name_column = TextColumn("{task.fields[name]}", table_column=Column(description, ratio=2))
    bar_column = BarColumn(bar_width=None, table_column=Column("", ratio=2))
    time_column = TimeElapsedColumn(table_column=Column("Elapsed", ratio=1))
    n_iters = MofNCompleteColumn(table_column=Column("Iteration"))

    step_size = TextColumn("{task.fields[step_size]:0.3f}", table_column=Column("Step", ratio=1))
    accept_rate = TextColumn(
        "{task.fields[accept_rate]:0.3f}", table_column=Column("Accept Rate", ratio=1)
    )

    objective_name = "Objective"
    obj_column = TextColumn(
        "{task.fields[f_value]:0.5f}", table_column=Column(objective_name, ratio=1)
    )

    columns = [name_column, bar_column, time_column, n_iters, step_size, accept_rate, obj_column]

    if use_jac:
        columns += [
            TextColumn("{task.fields[grad_norm]:0.5f}", table_column=Column("||grad||", ratio=1))
        ]
    if use_hess:
        columns += [
            TextColumn("{task.fields[hess_norm]:0.8f}", table_column=Column("||hess||", ratio=1))
        ]

    return ToggleableProgress(
        *columns,
        expand=False,
        disable=not progressbar,
        console=Console(file=sys.stderr, width=CONSOLE_WIDTH),
        transient=True,
    )


__all__ = ["initialize_progress_bar"]
