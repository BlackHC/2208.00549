__all__ = ['asclassdict', 'init_wandb', '__logging_initialized__', 'wandb_only', 'log2wandb']


import copy
import dataclasses
import functools

import wandb


__logging_initialized__ = None


# Helper to add class names to dicts (based on dataclasses.asdict)
def asclassdict(obj, *, dict_factory=dict):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage:

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts.
    """
    if not dataclasses._is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asclassdict_inner(obj, dict_factory)


def _asclassdict_inner(obj, dict_factory):
    if dataclasses._is_dataclass_instance(obj):
        result = []
        result.append(("Dataclass", f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"))
        for f in dataclasses.fields(obj):
            value = _asclassdict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asclassdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asclassdict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asclassdict_inner(k, dict_factory), _asclassdict_inner(v, dict_factory)) for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


def init_wandb(config, notes=None, project=None, entity=None):
    project = project or "dss"
    entity = entity or "oatml-andreas-kirsch"

    global __logging_initialized__

    wandb.init(
        project=project,
        entity=entity,
        config=asclassdict(config),
        save_code=True,
        job_type="experiment",
        notes=notes,
        # This codepath is currently broken.
        magic=False,
        mode="online",
    )

    __logging_initialized__ = True


def wandb_only(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global __logging_initialized__
        if __logging_initialized__ is None:
            wandb.init(mode="disabled")
            __logging_initialized__ = False
        f(*args, **kwargs)

    return wrapper


@wandb_only
def log2wandb(row, commit: bool):
    wandb.log(row, commit=commit)