import unittest
from pathlib import Path

from plag_in import recommend as recommend_module
from plag_in.recommend import RecommendationQuery, load_catalogue, recommend

_CATALOGUE_PATH = Path(__file__).parent.parent / "fixtures" / "catalogue.json"


class RecommendTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = load_catalogue(_CATALOGUE_PATH)

    def test_returns_at_most_three(self):
        query = RecommendationQuery(task="chat", available_memory_gb=64, preference="quality")
        options = recommend(query, self.catalogue)
        self.assertLessEqual(len(options), 3)

    def test_deterministic_ordering_for_repeated_calls(self):
        query = RecommendationQuery(task="chat", available_memory_gb=32, preference="quality")
        first = recommend(query, self.catalogue)
        second = recommend(query, self.catalogue)
        self.assertEqual([o["id"] for o in first], [o["id"] for o in second])

    def test_memory_bound_is_respected(self):
        query = RecommendationQuery(task="chat", available_memory_gb=7, preference="quality")
        options = recommend(query, self.catalogue)
        for option in options:
            self.assertLessEqual(option["memory_estimate_gb"], 7)

    def test_licence_constraint_filters(self):
        query = RecommendationQuery(task="chat", available_memory_gb=64, preference="quality", licence_constraint="apache-2.0")
        options = recommend(query, self.catalogue)
        for option in options:
            self.assertEqual(option["licence"], "apache-2.0")

    def test_unresolved_fields_are_labelled_not_hidden(self):
        query = RecommendationQuery(task="chat", available_memory_gb=64, preference="quality")
        options = recommend(query, self.catalogue)
        by_id = {o["id"]: o for o in options}
        if "cat-003" in by_id:
            self.assertTrue(by_id["cat-003"]["licence_unresolved"])
            self.assertIsNone(by_id["cat-003"]["download_command"])

    def test_module_has_no_download_function(self):
        names = dir(recommend_module)
        self.assertFalse(any("download" in name.lower() for name in names))

    def test_no_network_module_imported(self):
        import ast

        source = Path(recommend_module.__file__).read_text()
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("urllib", imported)
        self.assertNotIn("socket", imported)
        self.assertNotIn("requests", imported)


if __name__ == "__main__":
    unittest.main()
