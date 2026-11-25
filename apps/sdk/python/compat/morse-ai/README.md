# morse-ai (renamed to `amund`)

Morse is now **Amund**. The Python SDK is published as
[`amund`](https://pypi.org/project/amund/) and imported as `import amund`.

This `morse-ai` package is a compatibility shim. Installing it installs the
same version of `amund`, and `import morse_ai` keeps working as an alias of
`amund`: every module, class and function is the same object, so existing code
runs unchanged. Importing `morse_ai` emits a `DeprecationWarning`.

`MORSE_*` environment variables (for example `MORSE_API_KEY`) are still read.
The new names are `AMUND_*` (for example `AMUND_API_KEY`); when both are set,
the `AMUND_` value wins.

## Migrating

```bash
pip uninstall morse-ai
pip install amund
```

```python
import amund            # was: import morse_ai

amund.init()            # reads AMUND_API_KEY (or MORSE_API_KEY)
with amund.run("lead-qualifier") as r:
    ...
```

Docs: https://docs.amundai.com
