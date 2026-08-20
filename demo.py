"""Run the shipped benchmark and print the report.

    python demo.py
"""

from tutoreval import ablate, load_jsonl, render, render_ablation, run

DATA = "data/tutoring_benchmark_v1.jsonl"


def main() -> None:
    exchanges = load_jsonl(DATA)
    result = run(exchanges)
    print(render(result, show=4))
    print()
    print(render_ablation(ablate(exchanges)))
    print()
    print("Reading this report")
    print("-" * 64)
    print(
        "The pass rate describes the tutor responses in the dataset, not the\n"
        "harness. The precision and recall figures above describe the harness\n"
        "itself, measured against rows a person labelled by hand. Only the\n"
        "second pair tells you whether the first number means anything."
    )


if __name__ == "__main__":
    main()
