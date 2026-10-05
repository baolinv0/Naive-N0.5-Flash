"""The preregistered complete, discrete ISP recipe domain."""

import itertools
import json
import math


STRATEGIES = ('random', 'tpe', 'naive_scalar', 'naive_rich', 'naive_fast_slow')
RECIPE_DOMAINS = {
    'loss_family': ('original', 'mse', 'l1'),
    'optimizer': ('adam', 'adamw'),
    'learning_rate': (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    'weight_decay': (0, 1e-7, 1e-6, 1e-5, 1e-4),
}


def recipe_space():
    """Return fresh complete recipes in a stable order; never inherit a patch."""
    fields = tuple(RECIPE_DOMAINS)
    return [dict(zip(fields, values))
            for values in itertools.product(*(RECIPE_DOMAINS[k] for k in fields))]


def validate_space_recipe(recipe):
    """Require the exact four fields and domain; do not round or coerce input."""
    if not isinstance(recipe, dict) or set(recipe) != set(RECIPE_DOMAINS):
        raise ValueError('Recipe requires exactly the four complete recipe fields')
    result = {}
    for field, domain in RECIPE_DOMAINS.items():
        value = recipe[field]
        if field in ('learning_rate', 'weight_decay'):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{field} must be a finite number, not a boolean or string')
        elif type(value) is not str:
            raise ValueError(f'{field} must be a string')
        if value not in domain:
            raise ValueError(f'{field} is outside the preregistered discrete recipe domain')
        # Canonical domain members make 0 and 0.0 identical without permitting
        # string conversion or nearest-neighbor changes to a model proposal.
        result[field] = domain[domain.index(value)]
    return result


def recipe_key(recipe):
    return json.dumps(validate_space_recipe(recipe), sort_keys=True, separators=(',', ':'), allow_nan=False)
