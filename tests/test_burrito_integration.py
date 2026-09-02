import configparser
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = PROJECT_ROOT / "burrito"
WRAPPER_ROOT = INTEGRATION_ROOT / "wrapper"
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from adaptive_hrc_burrito import UpstreamPaths, verify_pins


class BurritoIntegrationTests(unittest.TestCase):
    def test_wrapper_does_not_claim_upstream_burrito_package_name(self):
        self.assertFalse((INTEGRATION_ROOT / "__init__.py").exists())
        self.assertTrue(
            (WRAPPER_ROOT / "adaptive_hrc_burrito" / "__init__.py").is_file()
        )

    def test_outer_and_nested_gitlinks_match_manifest(self):
        pins = json.loads((INTEGRATION_ROOT / "pins.json").read_text())
        modules = configparser.ConfigParser()
        modules.read(PROJECT_ROOT / ".gitmodules")
        outer = modules["submodule \"talents-zsc\""]
        self.assertEqual(outer["path"], "burrito/third_party/talents-zsc")
        self.assertEqual(outer["url"], pins["talents_zsc"]["url"])

        nested_file = (
            INTEGRATION_ROOT / pins["talents_zsc"]["path"] / ".gitmodules"
        )
        if not nested_file.exists():
            self.skipTest("TALENTS submodule is not initialized")
        nested = configparser.ConfigParser()
        nested.read(nested_file)
        overcooked = nested["submodule \"overcooked/overcooked_ai\""]
        self.assertEqual(overcooked["url"], pins["overcooked_ai"]["url"])

    def test_initialized_submodules_are_at_exact_pins(self):
        paths = UpstreamPaths.discover()
        if not paths.talents_root.exists():
            self.skipTest("TALENTS submodule is not initialized")
        pins = json.loads(paths.pins_file.read_text())
        self.assertEqual(
            verify_pins(paths),
            {
                "talents_zsc": pins["talents_zsc"]["commit"],
                "overcooked_ai": pins["overcooked_ai"]["commit"],
            },
        )
