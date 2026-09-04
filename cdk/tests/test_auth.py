import sys
from pathlib import Path

import unittest


AUTH_DIR = Path(__file__).parents[1] / "lambdas" / "presign_upload"
sys.path.insert(0, str(AUTH_DIR))

from auth import allowed_site_ids, resolve_site  # noqa: E402


def event_with_groups(groups: str) -> dict:
    return {"requestContext": {"authorizer": {"claims": {"cognito:groups": groups}}}}


class AuthTests(unittest.TestCase):
    def test_single_site_group_can_be_selected(self) -> None:
        event = event_with_groups("safeharbor-site-BCH")
        self.assertEqual(allowed_site_ids(event), {"bch"})
        self.assertEqual(resolve_site(event, "BCH"), ("BCH", ""))


    def test_user_cannot_select_unassigned_site(self) -> None:
        event = event_with_groups("safeharbor-site-BCH")
        site, error = resolve_site(event, "CHA")
        self.assertEqual(site, "")
        self.assertIn("not assigned", error)


    def test_multiple_sites_require_explicit_selection(self) -> None:
        event = event_with_groups("safeharbor-site-BCH safeharbor-site-CHA")
        self.assertEqual(
            resolve_site(event, ""),
            ("", "site_id is required when the user has multiple site assignments"),
        )


if __name__ == "__main__":
    unittest.main()
