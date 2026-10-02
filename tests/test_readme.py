import re

from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

# A block that cannot run as a standalone script, keyed by something only it contains.
NOT_STANDALONE = {
    "pytensor": "pytensor is an optional dependency",
    "multi_optimize(": 'the loky backend needs an `if __name__ == "__main__":` guard',
    "# TypeError:": "the block shows the error it raises, without the setup above it",
}


def readme_examples() -> list[tuple[str, str]]:
    """Every python block in the README, paired with the heading it appears under.

    The blocks are found first and the headings matched to them by position, because a
    block may contain comment lines that a heading pattern would otherwise match.
    """
    text = README.read_text()
    fences = [match.span() for match in re.finditer(r"```.*?```", text, re.DOTALL)]
    headings = [
        (match.start(), match.group(1))
        for match in re.finditer(r"^#+ (.*)$", text, flags=re.MULTILINE)
        if not any(start <= match.start() < end for start, end in fences)
    ]

    examples = []
    for block in re.finditer(r"```python\n(.*?)```", text, re.DOTALL):
        above = [title for start, title in headings if start < block.start()]
        examples.append((above[-1] if above else "intro", block.group(1)))

    return examples


def skip_reason(block: str) -> str | None:
    return next((reason for marker, reason in NOT_STANDALONE.items() if marker in block), None)


@pytest.mark.parametrize(
    "heading, block", readme_examples(), ids=[heading for heading, _ in readme_examples()]
)
def test_every_readme_example_runs(heading, block):
    """The README names configurations, keyword arguments and result attributes that no
    other test touches, so renaming one would otherwise be caught by a reader."""
    reason = skip_reason(block)
    if reason is not None:
        pytest.skip(reason)

    exec(compile(block, str(README), "exec"), {"__name__": "__readme__"})


def test_the_skipped_blocks_are_the_ones_expected():
    """A marker that stops matching would silently turn a skip into no coverage at all."""
    skipped = [block for _, block in readme_examples() if skip_reason(block) is not None]

    assert len(skipped) == len(NOT_STANDALONE)
