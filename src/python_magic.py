"""
Kompatibilitäts-Shim für `import python_magic`.

Unter Windows liefern sowohl `python-magic` als auch `python-magic-bin`
den Modulnamen `magic`. Dieses Shim stellt sicher, dass ein
`import python_magic` erfolgreich ist, indem es einfach das `magic`-Modul
weiterreicht.
"""

from magic import *  # noqa: F401,F403
