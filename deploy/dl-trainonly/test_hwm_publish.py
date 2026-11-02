"""Synthetic receiving-path and failure-injection fixtures; no private input or AWS access.

Test methods live in sibling mixins so this module remains the stable unittest entrypoint.
"""

import unittest

from _hwm_publish_fixture import PublisherFixture
from _hwm_publish_core import CoreCases
from _hwm_publish_campaign2 import Campaign2Cases
from _hwm_publish_catalog import CatalogCases


class PublisherTests(PublisherFixture, CoreCases, Campaign2Cases, CatalogCases):
    """All receiving-path cases collected under the original module/class name."""


if __name__ == "__main__":
    unittest.main()
