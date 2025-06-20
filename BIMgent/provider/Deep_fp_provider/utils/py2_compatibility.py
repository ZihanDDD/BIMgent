
class Py2CompatDict(dict):
    """A dictionary wrapper that provides Python 2 compatibility methods."""
    
    def iteritems(self):
        return self.items()
        
    def iterkeys(self):
        return self.keys()
        
    def itervalues(self):
        return self.values()

def wrap_dict(d):
    """Convert a standard dict to a Py2CompatDict."""
    if isinstance(d, dict) and not isinstance(d, Py2CompatDict):
        result = Py2CompatDict()
        for k, v in d.items():
            # Recursively convert nested dictionaries
            if isinstance(v, dict):
                result[k] = wrap_dict(v)
            else:
                result[k] = v
        return result
    return d
