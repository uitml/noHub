class NestedNamespace:
    def __init__(self, dct=None):
        self.keys = []
        if dct is not None:
            for key, value in dct.items():
                self._set_value(key, value)

    def _set_value(self, key, value):
        if "." in key:
            tokens = key.split(".")
            if hasattr(self, tokens[0]):
                sub = getattr(self, tokens[0])
            else:
                sub = NestedNamespace()
                setattr(self, tokens[0], sub)
                self.keys.append(tokens[0])

            sub._set_value(key=".".join(tokens[1:]), value=value)
        else:
            self.keys.append(key)
            setattr(self, key, value)

    def to_dict(self):
        out = {}
        for key in self.keys:
            value = getattr(self, key)
            if isinstance(value, NestedNamespace):
                sub_dict = value.to_dict()
                for sub_key, sub_value in sub_dict.items():
                    out[f"{key}.{sub_key}"] = sub_value
            else:
                out[key] = value
        return out

    def __str__(self):
        out = ""
        for key in self.keys:
            value = getattr(self, key)
            value_str = str(value)
            if "\n" in value_str:
                value_str = "\n".join([f"  {s}" for s in value_str.split("\n") if s != ""])
                out += f"{key}.\n{value_str}\n"
            else:
                out += f"{key} = {value_str}\n"
        return out

    def __repr__(self):
        return self.__str__()
