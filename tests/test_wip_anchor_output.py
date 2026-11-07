import csv

from WIP import WIPPlanner


def test_anchor_explanations_writer_accepts_tuple_feature_names(tmp_path):
    output_path = tmp_path / "anchors_explanations.csv"
    explanations = [{"speed": (1, 2), "temperature": (3, 4)}]

    WIPPlanner._write_anchor_explanations_csv(
        output_path,
        feature_names=("speed", "temperature"),
        req_names=("req_0", "req_1"),
        selected_samples=(0,),
        predictions=((1, 0),),
        explanations=explanations,
    )

    with output_path.open(newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [
        ["n_satisfied_reqs", "satisfied_reqs", "speed", "temperature"],
        ["1", "req_0", "(1, 2)", "(3, 4)"],
    ]
