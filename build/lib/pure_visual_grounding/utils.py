def ensure_utf8(value):
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    return value


def clean_metadata(metadata):
    return {k: ensure_utf8(v) for k, v in metadata.items()}

def clean_empty_fields_inplace(data):
    """
        recursively removes any empty strings/lists/dicts and None from json-like
        structure.
        @:param data Json Data
        @:returns True if element = empty after cleaning; false otherwise

    """
    if isinstance(data, dict):
        keys_to_delete = []
        for key, value in data.items():
            if clean_empty_fields_inplace(value):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del data[key]

    elif isinstance(data, list):
        items_to_keep = []
        for item in data:
            if not clean_empty_fields_inplace(item):
                items_to_keep.append(item)
        data[:] = items_to_keep

    return data in (None, "", [], {})