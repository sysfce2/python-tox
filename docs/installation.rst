Installation
============

As tool
-------

:pypi:`tox` is a CLI tool that needs a Python interpreter (version 3.9 or higher) to run. We recommend either
:pypi:`pipx` or :pypi:`uv` to install tox into an isolated environment. This has the added benefit that later you'll
be able to upgrade tox without affecting other parts of the system. We provide method for ``pip`` too here but we
discourage that path if you can:

.. tab:: uv

    .. code-block:: bash

        # install uv per https://docs.astral.sh/uv/#getting-started
        uv tool install tox
        tox --help

.. tab:: pipx

    .. code-block:: bash

        python -m pip install pipx-in-pipx --user
        pipx install tox
        tox --help

.. tab:: pip

    .. code-block:: bash

        python -m pip install --user tox
        python -m tox --help

    You can install it within the global Python interpreter itself (perhaps as a user package via the
    ``--user`` flag). Be cautious if you are using a Python installation that is managed by your operating system or
    another package manager. ``pip`` might not coordinate with those tools, and may leave your system in an inconsistent
    state. Note, if you go down this path you need to ensure pip is new enough per the subsections below

wheel
~~~~~

Installing tox via a wheel (default with pip) requires an installer that can understand the ``python-requires`` tag (see
:pep:`503`), with pip this is version ``9.0.0`` (released in November 2016). Furthermore, in case you're not installing
it via PyPI you need to use a mirror that correctly forwards the ``python-requires`` tag (notably the OpenStack mirrors
don't do this, or older :gh_repo:`devpi/devpi` versions - added with version ``4.7.0``).

.. _sdist:

sdist
~~~~~

When installing via a source distribution you need an installer that handles the :pep:`517` specification. In case of
``pip`` this is version ``18.0.0`` or later (released in July 2018). If you cannot upgrade your pip to support this you
need to ensure that the build requirements from :gh:`pyproject.toml <tox-dev/tox/blob/main/pyproject.toml>` are
satisfied before triggering the installation.

via ``setup.py``
----------------

We don't recommend and officially support this method. You should prefer using an installer that supports :pep:`517`
interface, such as pip ``19.0.0`` or later. That being said you might be able to still install a package via this method
if you satisfy build dependencies before calling the installation command (as described under :ref:`sdist`).

latest unreleased
-----------------

Installing an unreleased version is discouraged and should be only done for testing purposes. If you do so you'll need
a pip version of at least ``18.0.0`` and use the following command:


.. code-block:: bash

    pip install git+https://github.com/tox-dev/tox.git@main

.. _compatibility-requirements:

Python and OS Compatibility
---------------------------

tox works with the following Python interpreter implementations:

- `CPython <https://www.python.org/>`_ versions 3.9, 3.10, 3.11, 3.12, 3.13

This means tox works on the latest patch version of each of these minor versions. Previous patch versions are supported
on a best effort approach.
