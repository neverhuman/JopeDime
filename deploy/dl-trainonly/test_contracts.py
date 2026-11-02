"""CPU-only controls; synthetic objects never establish live GPU/S3 qualification."""

from test_contracts_budget import BudgetTests, Scratch
from test_contracts_output import OutputTests, bindings, context, pins

if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
