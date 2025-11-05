from typing import Dict

_CONTEXT_MODULE_REGISTRY = {}


def register_context_module(*names):
    """
    Decorator: registers a context module class under one or more names.
    
    Args:
        *names: One or more names to register the class under.
        
    Example:
        @register_context_module("default", "mlp")
        class MLPContextModule(BaseContextModule):
            pass
    """

    def decorator(cls):
        for name in names:
            _CONTEXT_MODULE_REGISTRY[name] = cls
        return cls

    return decorator


def get_context_module_cls(key: str) -> type:
    """
    Fetch the context module class for `key`. Raises if not found.
    
    Args:
        key: The name of the context module to retrieve.
        
    Returns:
        The context module class.
        
    Raises:
        ValueError: If the key is not found in the registry.
    """
    try:
        return _CONTEXT_MODULE_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown context module '{key}'. Available: {list(_CONTEXT_MODULE_REGISTRY.keys())}"
        )


def get_available_context_modules() -> list[str]:
    """
    Get a list of all available context module names.
    
    Returns:
        List of available context module names.
    """
    return list(_CONTEXT_MODULE_REGISTRY.keys())
