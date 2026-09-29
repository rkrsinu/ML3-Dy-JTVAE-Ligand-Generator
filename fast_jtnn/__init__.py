"""Lazy public exports for the ML3 JT-VAE implementation.

The upstream package eagerly imported modules that depend on
``fast_molopt.preprocess_prop``.  ``preprocess_prop`` also imports JT-VAE
modules, creating a circular import.  Lazy exports preserve the public API
without executing the dependency cycle during package initialization.
"""

_EXPORTS = {
    "MolTreeDataset": ("fast_jtnn.datautils_prop", "MolTreeDataset"),
    "set_batch_nodeID": ("fast_jtnn.datautils_prop", "set_batch_nodeID"),
    "JTMPN": ("fast_jtnn.jtmpn", "JTMPN"),
    "JTNNEncoder": ("fast_jtnn.jtnn_enc", "JTNNEncoder"),
    "JTNNVAE": ("fast_jtnn.jtnn_vae", "JTNNVAE"),
    "JTpropVAE": ("fast_jtnn.jtprop_vae", "JTpropVAE"),
    "MolTree": ("fast_jtnn.mol_tree", "MolTree"),
    "Vocab": ("fast_jtnn.vocab", "Vocab"),
    "MPN": ("fast_jtnn.mpn", "MPN"),
    "create_var": ("fast_jtnn.nnutils", "create_var"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
