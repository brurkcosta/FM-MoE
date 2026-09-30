_CACHE = {}

def get_cached(key, builder):
    """Constrói o modelo na primeira chamada e reaproveita nas seguintes."""
    if key not in _CACHE:
        print(f"[cache] carregando {key} (uma única vez)", flush=True)
        _CACHE[key] = builder()
    return _CACHE[key]