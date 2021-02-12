
'The Versioneer - like a rocketeer, but for versions.\n\nThe Versioneer\n==============\n\n* like a rocketeer, but for versions!\n* https://github.com/python-versioneer/python-versioneer\n* Brian Warner\n* License: Public Domain\n* Compatible with: Python 3.6, 3.7, 3.8, 3.9 and pypy3\n* [![Latest Version][pypi-image]][pypi-url]\n* [![Build Status][travis-image]][travis-url]\n\nThis is a tool for managing a recorded version number in distutils-based\npython projects. The goal is to remove the tedious and error-prone "update\nthe embedded version string" step from your release process. Making a new\nrelease should be as easy as recording a new tag in your version-control\nsystem, and maybe making new tarballs.\n\n\n## Quick Install\n\n* `pip install versioneer` to somewhere in your $PATH\n* add a `[versioneer]` section to your setup.cfg (see [Install](INSTALL.md))\n* run `versioneer install` in your source tree, commit the results\n* Verify version information with `python setup.py version`\n\n## Version Identifiers\n\nSource trees come from a variety of places:\n\n* a version-control system checkout (mostly used by developers)\n* a nightly tarball, produced by build automation\n* a snapshot tarball, produced by a web-based VCS browser, like github\'s\n  "tarball from tag" feature\n* a release tarball, produced by "setup.py sdist", distributed through PyPI\n\nWithin each source tree, the version identifier (either a string or a number,\nthis tool is format-agnostic) can come from a variety of places:\n\n* ask the VCS tool itself, e.g. "git describe" (for checkouts), which knows\n  about recent "tags" and an absolute revision-id\n* the name of the directory into which the tarball was unpacked\n* an expanded VCS keyword ($Id$, etc)\n* a `_version.py` created by some earlier build step\n\nFor released software, the version identifier is closely related to a VCS\ntag. Some projects use tag names that include more than just the version\nstring (e.g. "myproject-1.2" instead of just "1.2"), in which case the tool\nneeds to strip the tag prefix to extract the version identifier. For\nunreleased software (between tags), the version identifier should provide\nenough information to help developers recreate the same tree, while also\ngiving them an idea of roughly how old the tree is (after version 1.2, before\nversion 1.3). Many VCS systems can report a description that captures this,\nfor example `git describe --tags --dirty --always` reports things like\n"0.7-1-g574ab98-dirty" to indicate that the checkout is one revision past the\n0.7 tag, has a unique revision id of "574ab98", and is "dirty" (it has\nuncommitted changes).\n\nThe version identifier is used for multiple purposes:\n\n* to allow the module to self-identify its version: `myproject.__version__`\n* to choose a name and prefix for a \'setup.py sdist\' tarball\n\n## Theory of Operation\n\nVersioneer works by adding a special `_version.py` file into your source\ntree, where your `__init__.py` can import it. This `_version.py` knows how to\ndynamically ask the VCS tool for version information at import time.\n\n`_version.py` also contains `$Revision$` markers, and the installation\nprocess marks `_version.py` to have this marker rewritten with a tag name\nduring the `git archive` command. As a result, generated tarballs will\ncontain enough information to get the proper version.\n\nTo allow `setup.py` to compute a version too, a `versioneer.py` is added to\nthe top level of your source tree, next to `setup.py` and the `setup.cfg`\nthat configures it. This overrides several distutils/setuptools commands to\ncompute the version when invoked, and changes `setup.py build` and `setup.py\nsdist` to replace `_version.py` with a small static file that contains just\nthe generated version data.\n\n## Installation\n\nSee [INSTALL.md](./INSTALL.md) for detailed installation instructions.\n\n## Version-String Flavors\n\nCode which uses Versioneer can learn about its version string at runtime by\nimporting `_version` from your main `__init__.py` file and running the\n`get_versions()` function. From the "outside" (e.g. in `setup.py`), you can\nimport the top-level `versioneer.py` and run `get_versions()`.\n\nBoth functions return a dictionary with different flavors of version\ninformation:\n\n* `[\'version\']`: A condensed version string, rendered using the selected\n  style. This is the most commonly used value for the project\'s version\n  string. The default "pep440" style yields strings like `0.11`,\n  `0.11+2.g1076c97`, or `0.11+2.g1076c97.dirty`. See the "Styles" section\n  below for alternative styles.\n\n* `[\'full-revisionid\']`: detailed revision identifier. For Git, this is the\n  full SHA1 commit id, e.g. "1076c978a8d3cfc70f408fe5974aa6c092c949ac".\n\n* `[\'date\']`: Date and time of the latest `HEAD` commit. For Git, it is the\n  commit date in ISO 8601 format. This will be None if the date is not\n  available.\n\n* `[\'dirty\']`: a boolean, True if the tree has uncommitted changes. Note that\n  this is only accurate if run in a VCS checkout, otherwise it is likely to\n  be False or None\n\n* `[\'error\']`: if the version string could not be computed, this will be set\n  to a string describing the problem, otherwise it will be None. It may be\n  useful to throw an exception in setup.py if this is set, to avoid e.g.\n  creating tarballs with a version string of "unknown".\n\nSome variants are more useful than others. Including `full-revisionid` in a\nbug report should allow developers to reconstruct the exact code being tested\n(or indicate the presence of local changes that should be shared with the\ndevelopers). `version` is suitable for display in an "about" box or a CLI\n`--version` output: it can be easily compared against release notes and lists\nof bugs fixed in various releases.\n\nThe installer adds the following text to your `__init__.py` to place a basic\nversion in `YOURPROJECT.__version__`:\n\n    from ._version import get_versions\n    __version__ = get_versions()[\'version\']\n    del get_versions\n\n## Styles\n\nThe setup.cfg `style=` configuration controls how the VCS information is\nrendered into a version string.\n\nThe default style, "pep440", produces a PEP440-compliant string, equal to the\nun-prefixed tag name for actual releases, and containing an additional "local\nversion" section with more detail for in-between builds. For Git, this is\nTAG[+DISTANCE.gHEX[.dirty]] , using information from `git describe --tags\n--dirty --always`. For example "0.11+2.g1076c97.dirty" indicates that the\ntree is like the "1076c97" commit but has uncommitted changes (".dirty"), and\nthat this commit is two revisions ("+2") beyond the "0.11" tag. For released\nsoftware (exactly equal to a known tag), the identifier will only contain the\nstripped tag, e.g. "0.11".\n\nOther styles are available. See [details.md](details.md) in the Versioneer\nsource tree for descriptions.\n\n## Debugging\n\nVersioneer tries to avoid fatal errors: if something goes wrong, it will tend\nto return a version of "0+unknown". To investigate the problem, run `setup.py\nversion`, which will run the version-lookup code in a verbose mode, and will\ndisplay the full contents of `get_versions()` (including the `error` string,\nwhich may help identify what went wrong).\n\n## Known Limitations\n\nSome situations are known to cause problems for Versioneer. This details the\nmost significant ones. More can be found on Github\n[issues page](https://github.com/python-versioneer/python-versioneer/issues).\n\n### Subprojects\n\nVersioneer has limited support for source trees in which `setup.py` is not in\nthe root directory (e.g. `setup.py` and `.git/` are *not* siblings). The are\ntwo common reasons why `setup.py` might not be in the root:\n\n* Source trees which contain multiple subprojects, such as\n  [Buildbot](https://github.com/buildbot/buildbot), which contains both\n  "master" and "slave" subprojects, each with their own `setup.py`,\n  `setup.cfg`, and `tox.ini`. Projects like these produce multiple PyPI\n  distributions (and upload multiple independently-installable tarballs).\n* Source trees whose main purpose is to contain a C library, but which also\n  provide bindings to Python (and perhaps other languages) in subdirectories.\n\nVersioneer will look for `.git` in parent directories, and most operations\nshould get the right version string. However `pip` and `setuptools` have bugs\nand implementation details which frequently cause `pip install .` from a\nsubproject directory to fail to find a correct version string (so it usually\ndefaults to `0+unknown`).\n\n`pip install --editable .` should work correctly. `setup.py install` might\nwork too.\n\nPip-8.1.1 is known to have this problem, but hopefully it will get fixed in\nsome later version.\n\n[Bug #38](https://github.com/python-versioneer/python-versioneer/issues/38) is tracking\nthis issue. The discussion in\n[PR #61](https://github.com/python-versioneer/python-versioneer/pull/61) describes the\nissue from the Versioneer side in more detail.\n[pip PR#3176](https://github.com/pypa/pip/pull/3176) and\n[pip PR#3615](https://github.com/pypa/pip/pull/3615) contain work to improve\npip to let Versioneer work correctly.\n\nVersioneer-0.16 and earlier only looked for a `.git` directory next to the\n`setup.cfg`, so subprojects were completely unsupported with those releases.\n\n### Editable installs with setuptools <= 18.5\n\n`setup.py develop` and `pip install --editable .` allow you to install a\nproject into a virtualenv once, then continue editing the source code (and\ntest) without re-installing after every change.\n\n"Entry-point scripts" (`setup(entry_points={"console_scripts": ..})`) are a\nconvenient way to specify executable scripts that should be installed along\nwith the python package.\n\nThese both work as expected when using modern setuptools. When using\nsetuptools-18.5 or earlier, however, certain operations will cause\n`pkg_resources.DistributionNotFound` errors when running the entrypoint\nscript, which must be resolved by re-installing the package. This happens\nwhen the install happens with one version, then the egg_info data is\nregenerated while a different version is checked out. Many setup.py commands\ncause egg_info to be rebuilt (including `sdist`, `wheel`, and installing into\na different virtualenv), so this can be surprising.\n\n[Bug #83](https://github.com/python-versioneer/python-versioneer/issues/83) describes\nthis one, but upgrading to a newer version of setuptools should probably\nresolve it.\n\n\n## Updating Versioneer\n\nTo upgrade your project to a new release of Versioneer, do the following:\n\n* install the new Versioneer (`pip install -U versioneer` or equivalent)\n* edit `setup.cfg`, if necessary, to include any new configuration settings\n  indicated by the release notes. See [UPGRADING](./UPGRADING.md) for details.\n* re-run `versioneer install` in your source tree, to replace\n  `SRC/_version.py`\n* commit any changed files\n\n## Future Directions\n\nThis tool is designed to make it easily extended to other version-control\nsystems: all VCS-specific components are in separate directories like\nsrc/git/ . The top-level `versioneer.py` script is assembled from these\ncomponents by running make-versioneer.py . In the future, make-versioneer.py\nwill take a VCS name as an argument, and will construct a version of\n`versioneer.py` that is specific to the given VCS. It might also take the\nconfiguration arguments that are currently provided manually during\ninstallation by editing setup.py . Alternatively, it might go the other\ndirection and include code from all supported VCS systems, reducing the\nnumber of intermediate scripts.\n\n## Similar projects\n\n* [setuptools_scm](https://github.com/pypa/setuptools_scm/) - a non-vendored build-time\n  dependency\n* [minver](https://github.com/jbweston/miniver) - a lightweight reimplementation of\n  versioneer\n\n## License\n\nTo make Versioneer easier to embed, all its code is dedicated to the public\ndomain. The `_version.py` that it creates is also in the public domain.\nSpecifically, both are released under the Creative Commons "Public Domain\nDedication" license (CC0-1.0), as described in\nhttps://creativecommons.org/publicdomain/zero/1.0/ .\n\n[pypi-image]: https://img.shields.io/pypi/v/versioneer.svg\n[pypi-url]: https://pypi.python.org/pypi/versioneer/\n[travis-image]:\nhttps://img.shields.io/travis/com/python-versioneer/python-versioneer.svg\n[travis-url]: https://travis-ci.com/github/python-versioneer/python-versioneer\n\n'
import configparser
import errno
import json
import os
import re
import subprocess
import sys

class VersioneerConfig():
    'Container for Versioneer configuration parameters.'

def get_root():
    'Get the project root directory.\n\n    We require that all commands are run from the project root, i.e. the\n    directory that contains setup.py, setup.cfg, and versioneer.py .\n    '
    root = os.path.realpath(os.path.abspath(os.getcwd()))
    setup_py = os.path.join(root, 'setup.py')
    versioneer_py = os.path.join(root, 'versioneer.py')
    if (not (os.path.exists(setup_py) or os.path.exists(versioneer_py))):
        root = os.path.dirname(os.path.realpath(os.path.abspath(sys.argv[0])))
        setup_py = os.path.join(root, 'setup.py')
        versioneer_py = os.path.join(root, 'versioneer.py')
    if (not (os.path.exists(setup_py) or os.path.exists(versioneer_py))):
        err = "Versioneer was unable to run the project root directory. Versioneer requires setup.py to be executed from its immediate directory (like 'python setup.py COMMAND'), or in a way that lets it use sys.argv[0] to find the root (like 'python path/to/setup.py COMMAND')."
        raise VersioneerBadRootError(err)
    try:
        me = os.path.realpath(os.path.abspath(__file__))
        me_dir = os.path.normcase(os.path.splitext(me)[0])
        vsr_dir = os.path.normcase(os.path.splitext(versioneer_py)[0])
        if (me_dir != vsr_dir):
            print(('Warning: build in %s is using versioneer.py from %s' % (os.path.dirname(me), versioneer_py)))
    except NameError:
        pass
    return root

def get_config_from_root(root):
    'Read the project setup.cfg file to determine Versioneer config.'
    setup_cfg = os.path.join(root, 'setup.cfg')
    parser = configparser.ConfigParser()
    with open(setup_cfg) as f:
        parser.read_file(f)
    VCS = parser.get('versioneer', 'VCS')

    def get(parser, name):
        if parser.has_option('versioneer', name):
            return parser.get('versioneer', name)
        return None
    cfg = VersioneerConfig()
    cfg.VCS = VCS
    cfg.style = (get(parser, 'style') or '')
    cfg.versionfile_source = get(parser, 'versionfile_source')
    cfg.versionfile_build = get(parser, 'versionfile_build')
    cfg.tag_prefix = get(parser, 'tag_prefix')
    if (cfg.tag_prefix in ("''", '""')):
        cfg.tag_prefix = ''
    cfg.parentdir_prefix = get(parser, 'parentdir_prefix')
    cfg.verbose = get(parser, 'verbose')
    return cfg

class NotThisMethod(Exception):
    'Exception raised if a method is not valid for the current scenario.'
LONG_VERSION_PY = {}
HANDLERS = {}

def register_vcs_handler(vcs, method):
    'Create decorator to mark a method as the handler of a VCS.'

    def decorate(f):
        'Store f in HANDLERS[vcs][method].'
        if (vcs not in HANDLERS):
            HANDLERS[vcs] = {}
        HANDLERS[vcs][method] = f
        return f
    return decorate

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    'Call the given command(s).'
    assert isinstance(commands, list)
    p = None
    for c in commands:
        try:
            dispcmd = str(([c] + args))
            p = subprocess.Popen(([c] + args), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=(subprocess.PIPE if hide_stderr else None))
            break
        except OSError:
            e = sys.exc_info()[1]
            if (e.errno == errno.ENOENT):
                continue
            if verbose:
                print(('unable to run %s' % dispcmd))
                print(e)
            return (None, None)
    else:
        if verbose:
            print(f'unable to find command, tried {commands}')
        return (None, None)
    stdout = p.communicate()[0].strip().decode()
    if (p.returncode != 0):
        if verbose:
            print(('unable to run %s (error)' % dispcmd))
            print(('stdout was %s' % stdout))
        return (None, p.returncode)
    return (stdout, p.returncode)
LONG_VERSION_PY['git'] = '\n# This file helps to compute a version number in source trees obtained from\n# git-archive tarball (such as those provided by githubs download-from-tag\n# feature). Distribution tarballs (built by setup.py sdist) and build\n# directories (produced by setup.py build) will contain a much shorter file\n# that just contains the computed version number.\n\n# This file is released into the public domain. Generated by\n# versioneer-0.19 (https://github.com/python-versioneer/python-versioneer)\n\n"""Git implementation of _version.py."""\n\nimport errno\nimport os\nimport re\nimport subprocess\nimport sys\n\n\ndef get_keywords():\n    """Get the keywords needed to look up the version information."""\n    # these strings will be replaced by git during git-archive.\n    # setup.py/versioneer.py will grep for the variable names, so they must\n    # each be defined on a line of their own. _version.py will just call\n    # get_keywords().\n    git_refnames = "%(DOLLAR)sFormat:%%d%(DOLLAR)s"\n    git_full = "%(DOLLAR)sFormat:%%H%(DOLLAR)s"\n    git_date = "%(DOLLAR)sFormat:%%ci%(DOLLAR)s"\n    keywords = {"refnames": git_refnames, "full": git_full, "date": git_date}\n    return keywords\n\n\nclass VersioneerConfig:\n    """Container for Versioneer configuration parameters."""\n\n\ndef get_config():\n    """Create, populate and return the VersioneerConfig() object."""\n    # these strings are filled in when \'setup.py versioneer\' creates\n    # _version.py\n    cfg = VersioneerConfig()\n    cfg.VCS = "git"\n    cfg.style = "%(STYLE)s"\n    cfg.tag_prefix = "%(TAG_PREFIX)s"\n    cfg.parentdir_prefix = "%(PARENTDIR_PREFIX)s"\n    cfg.versionfile_source = "%(VERSIONFILE_SOURCE)s"\n    cfg.verbose = False\n    return cfg\n\n\nclass NotThisMethod(Exception):\n    """Exception raised if a method is not valid for the current scenario."""\n\n\nLONG_VERSION_PY = {}\nHANDLERS = {}\n\n\ndef register_vcs_handler(vcs, method):  # decorator\n    """Create decorator to mark a method as the handler of a VCS."""\n    def decorate(f):\n        """Store f in HANDLERS[vcs][method]."""\n        if vcs not in HANDLERS:\n            HANDLERS[vcs] = {}\n        HANDLERS[vcs][method] = f\n        return f\n    return decorate\n\n\ndef run_command(commands, args, cwd=None, verbose=False, hide_stderr=False,\n                env=None):\n    """Call the given command(s)."""\n    assert isinstance(commands, list)\n    p = None\n    for c in commands:\n        try:\n            dispcmd = str([c] + args)\n            # remember shell=False, so use git.cmd on windows, not just git\n            p = subprocess.Popen([c] + args, cwd=cwd, env=env,\n                                 stdout=subprocess.PIPE,\n                                 stderr=(subprocess.PIPE if hide_stderr\n                                         else None))\n            break\n        except EnvironmentError:\n            e = sys.exc_info()[1]\n            if e.errno == errno.ENOENT:\n                continue\n            if verbose:\n                print("unable to run %%s" %% dispcmd)\n                print(e)\n            return None, None\n    else:\n        if verbose:\n            print("unable to find command, tried %%s" %% (commands,))\n        return None, None\n    stdout = p.communicate()[0].strip().decode()\n    if p.returncode != 0:\n        if verbose:\n            print("unable to run %%s (error)" %% dispcmd)\n            print("stdout was %%s" %% stdout)\n        return None, p.returncode\n    return stdout, p.returncode\n\n\ndef versions_from_parentdir(parentdir_prefix, root, verbose):\n    """Try to determine the version from the parent directory name.\n\n    Source tarballs conventionally unpack into a directory that includes both\n    the project name and a version string. We will also support searching up\n    two directory levels for an appropriately named parent directory\n    """\n    rootdirs = []\n\n    for i in range(3):\n        dirname = os.path.basename(root)\n        if dirname.startswith(parentdir_prefix):\n            return {"version": dirname[len(parentdir_prefix):],\n                    "full-revisionid": None,\n                    "dirty": False, "error": None, "date": None}\n        else:\n            rootdirs.append(root)\n            root = os.path.dirname(root)  # up a level\n\n    if verbose:\n        print("Tried directories %%s but none started with prefix %%s" %%\n              (str(rootdirs), parentdir_prefix))\n    raise NotThisMethod("rootdir doesn\'t start with parentdir_prefix")\n\n\n@register_vcs_handler("git", "get_keywords")\ndef git_get_keywords(versionfile_abs):\n    """Extract version information from the given file."""\n    # the code embedded in _version.py can just fetch the value of these\n    # keywords. When used from setup.py, we don\'t want to import _version.py,\n    # so we do it with a regexp instead. This function is not used from\n    # _version.py.\n    keywords = {}\n    try:\n        f = open(versionfile_abs, "r")\n        for line in f.readlines():\n            if line.strip().startswith("git_refnames ="):\n                mo = re.search(r\'=\\s*"(.*)"\', line)\n                if mo:\n                    keywords["refnames"] = mo.group(1)\n            if line.strip().startswith("git_full ="):\n                mo = re.search(r\'=\\s*"(.*)"\', line)\n                if mo:\n                    keywords["full"] = mo.group(1)\n            if line.strip().startswith("git_date ="):\n                mo = re.search(r\'=\\s*"(.*)"\', line)\n                if mo:\n                    keywords["date"] = mo.group(1)\n        f.close()\n    except EnvironmentError:\n        pass\n    return keywords\n\n\n@register_vcs_handler("git", "keywords")\ndef git_versions_from_keywords(keywords, tag_prefix, verbose):\n    """Get version information from git keywords."""\n    if not keywords:\n        raise NotThisMethod("no keywords at all, weird")\n    date = keywords.get("date")\n    if date is not None:\n        # Use only the last line.  Previous lines may contain GPG signature\n        # information.\n        date = date.splitlines()[-1]\n\n        # git-2.2.0 added "%%cI", which expands to an ISO-8601 -compliant\n        # datestamp. However we prefer "%%ci" (which expands to an "ISO-8601\n        # -like" string, which we must then edit to make compliant), because\n        # it\'s been around since git-1.5.3, and it\'s too difficult to\n        # discover which version we\'re using, or to work around using an\n        # older one.\n        date = date.strip().replace(" ", "T", 1).replace(" ", "", 1)\n    refnames = keywords["refnames"].strip()\n    if refnames.startswith("$Format"):\n        if verbose:\n            print("keywords are unexpanded, not using")\n        raise NotThisMethod("unexpanded keywords, not a git-archive tarball")\n    refs = set([r.strip() for r in refnames.strip("()").split(",")])\n    # starting in git-1.8.3, tags are listed as "tag: foo-1.0" instead of\n    # just "foo-1.0". If we see a "tag: " prefix, prefer those.\n    TAG = "tag: "\n    tags = set([r[len(TAG):] for r in refs if r.startswith(TAG)])\n    if not tags:\n        # Either we\'re using git < 1.8.3, or there really are no tags. We use\n        # a heuristic: assume all version tags have a digit. The old git %%d\n        # expansion behaves like git log --decorate=short and strips out the\n        # refs/heads/ and refs/tags/ prefixes that would let us distinguish\n        # between branches and tags. By ignoring refnames without digits, we\n        # filter out many common branch names like "release" and\n        # "stabilization", as well as "HEAD" and "master".\n        tags = set([r for r in refs if re.search(r\'\\d\', r)])\n        if verbose:\n            print("discarding \'%%s\', no digits" %% ",".join(refs - tags))\n    if verbose:\n        print("likely tags: %%s" %% ",".join(sorted(tags)))\n    for ref in sorted(tags):\n        # sorting will prefer e.g. "2.0" over "2.0rc1"\n        if ref.startswith(tag_prefix):\n            r = ref[len(tag_prefix):]\n            if verbose:\n                print("picking %%s" %% r)\n            return {"version": r,\n                    "full-revisionid": keywords["full"].strip(),\n                    "dirty": False, "error": None,\n                    "date": date}\n    # no suitable tags, so version is "0+unknown", but full hex is still there\n    if verbose:\n        print("no suitable tags, using unknown + full revision id")\n    return {"version": "0+unknown",\n            "full-revisionid": keywords["full"].strip(),\n            "dirty": False, "error": "no suitable tags", "date": None}\n\n\n@register_vcs_handler("git", "pieces_from_vcs")\ndef git_pieces_from_vcs(tag_prefix, root, verbose, run_command=run_command):\n    """Get version from \'git describe\' in the root of the source tree.\n\n    This only gets called if the git-archive \'subst\' keywords were *not*\n    expanded, and _version.py hasn\'t already been rewritten with a short\n    version string, meaning we\'re inside a checked out source tree.\n    """\n    GITS = ["git"]\n    if sys.platform == "win32":\n        GITS = ["git.cmd", "git.exe"]\n\n    out, rc = run_command(GITS, ["rev-parse", "--git-dir"], cwd=root,\n                          hide_stderr=True)\n    if rc != 0:\n        if verbose:\n            print("Directory %%s not under git control" %% root)\n        raise NotThisMethod("\'git rev-parse --git-dir\' returned error")\n\n    # if there is a tag matching tag_prefix, this yields TAG-NUM-gHEX[-dirty]\n    # if there isn\'t one, this yields HEX[-dirty] (no NUM)\n    describe_out, rc = run_command(GITS, ["describe", "--tags", "--dirty",\n                                          "--always", "--long",\n                                          "--match", "%%s*" %% tag_prefix],\n                                   cwd=root)\n    # --long was added in git-1.5.5\n    if describe_out is None:\n        raise NotThisMethod("\'git describe\' failed")\n    describe_out = describe_out.strip()\n    full_out, rc = run_command(GITS, ["rev-parse", "HEAD"], cwd=root)\n    if full_out is None:\n        raise NotThisMethod("\'git rev-parse\' failed")\n    full_out = full_out.strip()\n\n    pieces = {}\n    pieces["long"] = full_out\n    pieces["short"] = full_out[:7]  # maybe improved later\n    pieces["error"] = None\n\n    # parse describe_out. It will be like TAG-NUM-gHEX[-dirty] or HEX[-dirty]\n    # TAG might have hyphens.\n    git_describe = describe_out\n\n    # look for -dirty suffix\n    dirty = git_describe.endswith("-dirty")\n    pieces["dirty"] = dirty\n    if dirty:\n        git_describe = git_describe[:git_describe.rindex("-dirty")]\n\n    # now we have TAG-NUM-gHEX or HEX\n\n    if "-" in git_describe:\n        # TAG-NUM-gHEX\n        mo = re.search(r\'^(.+)-(\\d+)-g([0-9a-f]+)$\', git_describe)\n        if not mo:\n            # unparseable. Maybe git-describe is misbehaving?\n            pieces["error"] = ("unable to parse git-describe output: \'%%s\'"\n                               %% describe_out)\n            return pieces\n\n        # tag\n        full_tag = mo.group(1)\n        if not full_tag.startswith(tag_prefix):\n            if verbose:\n                fmt = "tag \'%%s\' doesn\'t start with prefix \'%%s\'"\n                print(fmt %% (full_tag, tag_prefix))\n            pieces["error"] = ("tag \'%%s\' doesn\'t start with prefix \'%%s\'"\n                               %% (full_tag, tag_prefix))\n            return pieces\n        pieces["closest-tag"] = full_tag[len(tag_prefix):]\n\n        # distance: number of commits since tag\n        pieces["distance"] = int(mo.group(2))\n\n        # commit: short hex revision ID\n        pieces["short"] = mo.group(3)\n\n    else:\n        # HEX: no tags\n        pieces["closest-tag"] = None\n        count_out, rc = run_command(GITS, ["rev-list", "HEAD", "--count"],\n                                    cwd=root)\n        pieces["distance"] = int(count_out)  # total number of commits\n\n    # commit date: see ISO-8601 comment in git_versions_from_keywords()\n    date = run_command(GITS, ["show", "-s", "--format=%%ci", "HEAD"],\n                       cwd=root)[0].strip()\n    # Use only the last line.  Previous lines may contain GPG signature\n    # information.\n    date = date.splitlines()[-1]\n    pieces["date"] = date.strip().replace(" ", "T", 1).replace(" ", "", 1)\n\n    return pieces\n\n\ndef plus_or_dot(pieces):\n    """Return a + if we don\'t already have one, else return a ."""\n    if "+" in pieces.get("closest-tag", ""):\n        return "."\n    return "+"\n\n\ndef render_pep440(pieces):\n    """Build up version string, with post-release "local version identifier".\n\n    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you\n    get a tagged build and then dirty it, you\'ll get TAG+0.gHEX.dirty\n\n    Exceptions:\n    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        if pieces["distance"] or pieces["dirty"]:\n            rendered += plus_or_dot(pieces)\n            rendered += "%%d.g%%s" %% (pieces["distance"], pieces["short"])\n            if pieces["dirty"]:\n                rendered += ".dirty"\n    else:\n        # exception #1\n        rendered = "0+untagged.%%d.g%%s" %% (pieces["distance"],\n                                          pieces["short"])\n        if pieces["dirty"]:\n            rendered += ".dirty"\n    return rendered\n\n\ndef render_pep440_pre(pieces):\n    """TAG[.post0.devDISTANCE] -- No -dirty.\n\n    Exceptions:\n    1: no tags. 0.post0.devDISTANCE\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        if pieces["distance"]:\n            rendered += ".post0.dev%%d" %% pieces["distance"]\n    else:\n        # exception #1\n        rendered = "0.post0.dev%%d" %% pieces["distance"]\n    return rendered\n\n\ndef render_pep440_post(pieces):\n    """TAG[.postDISTANCE[.dev0]+gHEX] .\n\n    The ".dev0" means dirty. Note that .dev0 sorts backwards\n    (a dirty tree will appear "older" than the corresponding clean one),\n    but you shouldn\'t be releasing software with -dirty anyways.\n\n    Exceptions:\n    1: no tags. 0.postDISTANCE[.dev0]\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        if pieces["distance"] or pieces["dirty"]:\n            rendered += ".post%%d" %% pieces["distance"]\n            if pieces["dirty"]:\n                rendered += ".dev0"\n            rendered += plus_or_dot(pieces)\n            rendered += "g%%s" %% pieces["short"]\n    else:\n        # exception #1\n        rendered = "0.post%%d" %% pieces["distance"]\n        if pieces["dirty"]:\n            rendered += ".dev0"\n        rendered += "+g%%s" %% pieces["short"]\n    return rendered\n\n\ndef render_pep440_old(pieces):\n    """TAG[.postDISTANCE[.dev0]] .\n\n    The ".dev0" means dirty.\n\n    Exceptions:\n    1: no tags. 0.postDISTANCE[.dev0]\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        if pieces["distance"] or pieces["dirty"]:\n            rendered += ".post%%d" %% pieces["distance"]\n            if pieces["dirty"]:\n                rendered += ".dev0"\n    else:\n        # exception #1\n        rendered = "0.post%%d" %% pieces["distance"]\n        if pieces["dirty"]:\n            rendered += ".dev0"\n    return rendered\n\n\ndef render_git_describe(pieces):\n    """TAG[-DISTANCE-gHEX][-dirty].\n\n    Like \'git describe --tags --dirty --always\'.\n\n    Exceptions:\n    1: no tags. HEX[-dirty]  (note: no \'g\' prefix)\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        if pieces["distance"]:\n            rendered += "-%%d-g%%s" %% (pieces["distance"], pieces["short"])\n    else:\n        # exception #1\n        rendered = pieces["short"]\n    if pieces["dirty"]:\n        rendered += "-dirty"\n    return rendered\n\n\ndef render_git_describe_long(pieces):\n    """TAG-DISTANCE-gHEX[-dirty].\n\n    Like \'git describe --tags --dirty --always -long\'.\n    The distance/hash is unconditional.\n\n    Exceptions:\n    1: no tags. HEX[-dirty]  (note: no \'g\' prefix)\n    """\n    if pieces["closest-tag"]:\n        rendered = pieces["closest-tag"]\n        rendered += "-%%d-g%%s" %% (pieces["distance"], pieces["short"])\n    else:\n        # exception #1\n        rendered = pieces["short"]\n    if pieces["dirty"]:\n        rendered += "-dirty"\n    return rendered\n\n\ndef render(pieces, style):\n    """Render the given version pieces into the requested style."""\n    if pieces["error"]:\n        return {"version": "unknown",\n                "full-revisionid": pieces.get("long"),\n                "dirty": None,\n                "error": pieces["error"],\n                "date": None}\n\n    if not style or style == "default":\n        style = "pep440"  # the default\n\n    if style == "pep440":\n        rendered = render_pep440(pieces)\n    elif style == "pep440-pre":\n        rendered = render_pep440_pre(pieces)\n    elif style == "pep440-post":\n        rendered = render_pep440_post(pieces)\n    elif style == "pep440-old":\n        rendered = render_pep440_old(pieces)\n    elif style == "git-describe":\n        rendered = render_git_describe(pieces)\n    elif style == "git-describe-long":\n        rendered = render_git_describe_long(pieces)\n    else:\n        raise ValueError("unknown style \'%%s\'" %% style)\n\n    return {"version": rendered, "full-revisionid": pieces["long"],\n            "dirty": pieces["dirty"], "error": None,\n            "date": pieces.get("date")}\n\n\ndef get_versions():\n    """Get version information or return default if unable to do so."""\n    # I am in _version.py, which lives at ROOT/VERSIONFILE_SOURCE. If we have\n    # __file__, we can work backwards from there to the root. Some\n    # py2exe/bbfreeze/non-CPython implementations don\'t do __file__, in which\n    # case we can only use expanded keywords.\n\n    cfg = get_config()\n    verbose = cfg.verbose\n\n    try:\n        return git_versions_from_keywords(get_keywords(), cfg.tag_prefix,\n                                          verbose)\n    except NotThisMethod:\n        pass\n\n    try:\n        root = os.path.realpath(__file__)\n        # versionfile_source is the relative path from the top of the source\n        # tree (where the .git directory might live) to this file. Invert\n        # this to find the root from __file__.\n        for i in cfg.versionfile_source.split(\'/\'):\n            root = os.path.dirname(root)\n    except NameError:\n        return {"version": "0+unknown", "full-revisionid": None,\n                "dirty": None,\n                "error": "unable to find root of source tree",\n                "date": None}\n\n    try:\n        pieces = git_pieces_from_vcs(cfg.tag_prefix, root, verbose)\n        return render(pieces, cfg.style)\n    except NotThisMethod:\n        pass\n\n    try:\n        if cfg.parentdir_prefix:\n            return versions_from_parentdir(cfg.parentdir_prefix, root, verbose)\n    except NotThisMethod:\n        pass\n\n    return {"version": "0+unknown", "full-revisionid": None,\n            "dirty": None,\n            "error": "unable to compute version", "date": None}\n'

@register_vcs_handler('git', 'get_keywords')
def git_get_keywords(versionfile_abs):
    'Extract version information from the given file.'
    keywords = {}
    try:
        f = open(versionfile_abs)
        for line in f.readlines():
            if line.strip().startswith('git_refnames ='):
                mo = re.search('=\\s*"(.*)"', line)
                if mo:
                    keywords['refnames'] = mo.group(1)
            if line.strip().startswith('git_full ='):
                mo = re.search('=\\s*"(.*)"', line)
                if mo:
                    keywords['full'] = mo.group(1)
            if line.strip().startswith('git_date ='):
                mo = re.search('=\\s*"(.*)"', line)
                if mo:
                    keywords['date'] = mo.group(1)
        f.close()
    except OSError:
        pass
    return keywords

@register_vcs_handler('git', 'keywords')
def git_versions_from_keywords(keywords, tag_prefix, verbose):
    'Get version information from git keywords.'
    if (not keywords):
        raise NotThisMethod('no keywords at all, weird')
    date = keywords.get('date')
    if (date is not None):
        date = date.splitlines()[(- 1)]
        date = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    refnames = keywords['refnames'].strip()
    if refnames.startswith('$Format'):
        if verbose:
            print('keywords are unexpanded, not using')
        raise NotThisMethod('unexpanded keywords, not a git-archive tarball')
    refs = {r.strip() for r in refnames.strip('()').split(',')}
    TAG = 'tag: '
    tags = {r[len(TAG):] for r in refs if r.startswith(TAG)}
    if (not tags):
        tags = {r for r in refs if re.search('\\d', r)}
        if verbose:
            print(("discarding '%s', no digits" % ','.join((refs - tags))))
    if verbose:
        print(('likely tags: %s' % ','.join(sorted(tags))))
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r = ref[len(tag_prefix):]
            if verbose:
                print(('picking %s' % r))
            return {'version': r, 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': None, 'date': date}
    if verbose:
        print('no suitable tags, using unknown + full revision id')
    return {'version': '0+unknown', 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': 'no suitable tags', 'date': None}

@register_vcs_handler('git', 'pieces_from_vcs')
def git_pieces_from_vcs(tag_prefix, root, verbose, run_command=run_command):
    "Get version from 'git describe' in the root of the source tree.\n\n    This only gets called if the git-archive 'subst' keywords were *not*\n    expanded, and _version.py hasn't already been rewritten with a short\n    version string, meaning we're inside a checked out source tree.\n    "
    GITS = ['git']
    if (sys.platform == 'win32'):
        GITS = ['git.cmd', 'git.exe']
    (out, rc) = run_command(GITS, ['rev-parse', '--git-dir'], cwd=root, hide_stderr=True)
    if (rc != 0):
        if verbose:
            print(('Directory %s not under git control' % root))
        raise NotThisMethod("'git rev-parse --git-dir' returned error")
    (describe_out, rc) = run_command(GITS, ['describe', '--tags', '--dirty', '--always', '--long', '--match', ('%s*' % tag_prefix)], cwd=root)
    if (describe_out is None):
        raise NotThisMethod("'git describe' failed")
    describe_out = describe_out.strip()
    (full_out, rc) = run_command(GITS, ['rev-parse', 'HEAD'], cwd=root)
    if (full_out is None):
        raise NotThisMethod("'git rev-parse' failed")
    full_out = full_out.strip()
    pieces = {}
    pieces['long'] = full_out
    pieces['short'] = full_out[:7]
    pieces['error'] = None
    git_describe = describe_out
    dirty = git_describe.endswith('-dirty')
    pieces['dirty'] = dirty
    if dirty:
        git_describe = git_describe[:git_describe.rindex('-dirty')]
    if ('-' in git_describe):
        mo = re.search('^(.+)-(\\d+)-g([0-9a-f]+)$', git_describe)
        if (not mo):
            pieces['error'] = ("unable to parse git-describe output: '%s'" % describe_out)
            return pieces
        full_tag = mo.group(1)
        if (not full_tag.startswith(tag_prefix)):
            if verbose:
                fmt = "tag '%s' doesn't start with prefix '%s'"
                print((fmt % (full_tag, tag_prefix)))
            pieces['error'] = "tag '{}' doesn't start with prefix '{}'".format(full_tag, tag_prefix)
            return pieces
        pieces['closest-tag'] = full_tag[len(tag_prefix):]
        pieces['distance'] = int(mo.group(2))
        pieces['short'] = mo.group(3)
    else:
        pieces['closest-tag'] = None
        (count_out, rc) = run_command(GITS, ['rev-list', 'HEAD', '--count'], cwd=root)
        pieces['distance'] = int(count_out)
    date = run_command(GITS, ['show', '-s', '--format=%ci', 'HEAD'], cwd=root)[0].strip()
    date = date.splitlines()[(- 1)]
    pieces['date'] = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    return pieces

def do_vcs_install(manifest_in, versionfile_source, ipy):
    'Git-specific installation logic for Versioneer.\n\n    For Git, this means creating/changing .gitattributes to mark _version.py\n    for export-subst keyword substitution.\n    '
    GITS = ['git']
    if (sys.platform == 'win32'):
        GITS = ['git.cmd', 'git.exe']
    files = [manifest_in, versionfile_source]
    if ipy:
        files.append(ipy)
    try:
        me = __file__
        if (me.endswith('.pyc') or me.endswith('.pyo')):
            me = (os.path.splitext(me)[0] + '.py')
        versioneer_file = os.path.relpath(me)
    except NameError:
        versioneer_file = 'versioneer.py'
    files.append(versioneer_file)
    present = False
    try:
        f = open('.gitattributes')
        for line in f.readlines():
            if line.strip().startswith(versionfile_source):
                if ('export-subst' in line.strip().split()[1:]):
                    present = True
        f.close()
    except OSError:
        pass
    if (not present):
        f = open('.gitattributes', 'a+')
        f.write(('%s export-subst\n' % versionfile_source))
        f.close()
        files.append('.gitattributes')
    run_command(GITS, (['add', '--'] + files))

def versions_from_parentdir(parentdir_prefix, root, verbose):
    'Try to determine the version from the parent directory name.\n\n    Source tarballs conventionally unpack into a directory that includes both\n    the project name and a version string. We will also support searching up\n    two directory levels for an appropriately named parent directory\n    '
    rootdirs = []
    for i in range(3):
        dirname = os.path.basename(root)
        if dirname.startswith(parentdir_prefix):
            return {'version': dirname[len(parentdir_prefix):], 'full-revisionid': None, 'dirty': False, 'error': None, 'date': None}
        else:
            rootdirs.append(root)
            root = os.path.dirname(root)
    if verbose:
        print(('Tried directories %s but none started with prefix %s' % (str(rootdirs), parentdir_prefix)))
    raise NotThisMethod("rootdir doesn't start with parentdir_prefix")
SHORT_VERSION_PY = "\n# This file was generated by 'versioneer.py' (0.19) from\n# revision-control system data, or from the parent directory name of an\n# unpacked source archive. Distribution tarballs contain a pre-generated copy\n# of this file.\n\nimport json\n\nversion_json = '''\n%s\n'''  # END VERSION_JSON\n\n\ndef get_versions():\n    return json.loads(version_json)\n"

def versions_from_file(filename):
    'Try to determine the version from _version.py if present.'
    try:
        with open(filename) as f:
            contents = f.read()
    except OSError:
        raise NotThisMethod('unable to read _version.py')
    mo = re.search("version_json = '''\\n(.*)'''  # END VERSION_JSON", contents, (re.M | re.S))
    if (not mo):
        mo = re.search("version_json = '''\\r\\n(.*)'''  # END VERSION_JSON", contents, (re.M | re.S))
    if (not mo):
        raise NotThisMethod('no version_json in _version.py')
    return json.loads(mo.group(1))

def write_to_version_file(filename, versions):
    'Write the given version number to the given _version.py file.'
    os.unlink(filename)
    contents = json.dumps(versions, sort_keys=True, indent=1, separators=(',', ': '))
    with open(filename, 'w') as f:
        f.write((SHORT_VERSION_PY % contents))
    print("set {} to '{}'".format(filename, versions['version']))

def plus_or_dot(pieces):
    "Return a + if we don't already have one, else return a ."
    if ('+' in pieces.get('closest-tag', '')):
        return '.'
    return '+'

def render_pep440(pieces):
    'Build up version string, with post-release "local version identifier".\n\n    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you\n    get a tagged build and then dirty it, you\'ll get TAG+0.gHEX.dirty\n\n    Exceptions:\n    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]\n    '
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if (pieces['distance'] or pieces['dirty']):
            rendered += plus_or_dot(pieces)
            rendered += ('%d.g%s' % (pieces['distance'], pieces['short']))
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = ('0+untagged.%d.g%s' % (pieces['distance'], pieces['short']))
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered

def render_pep440_pre(pieces):
    'TAG[.post0.devDISTANCE] -- No -dirty.\n\n    Exceptions:\n    1: no tags. 0.post0.devDISTANCE\n    '
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance']:
            rendered += ('.post0.dev%d' % pieces['distance'])
    else:
        rendered = ('0.post0.dev%d' % pieces['distance'])
    return rendered

def render_pep440_post(pieces):
    'TAG[.postDISTANCE[.dev0]+gHEX] .\n\n    The ".dev0" means dirty. Note that .dev0 sorts backwards\n    (a dirty tree will appear "older" than the corresponding clean one),\n    but you shouldn\'t be releasing software with -dirty anyways.\n\n    Exceptions:\n    1: no tags. 0.postDISTANCE[.dev0]\n    '
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if (pieces['distance'] or pieces['dirty']):
            rendered += ('.post%d' % pieces['distance'])
            if pieces['dirty']:
                rendered += '.dev0'
            rendered += plus_or_dot(pieces)
            rendered += ('g%s' % pieces['short'])
    else:
        rendered = ('0.post%d' % pieces['distance'])
        if pieces['dirty']:
            rendered += '.dev0'
        rendered += ('+g%s' % pieces['short'])
    return rendered

def render_pep440_old(pieces):
    'TAG[.postDISTANCE[.dev0]] .\n\n    The ".dev0" means dirty.\n\n    Exceptions:\n    1: no tags. 0.postDISTANCE[.dev0]\n    '
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if (pieces['distance'] or pieces['dirty']):
            rendered += ('.post%d' % pieces['distance'])
            if pieces['dirty']:
                rendered += '.dev0'
    else:
        rendered = ('0.post%d' % pieces['distance'])
        if pieces['dirty']:
            rendered += '.dev0'
    return rendered

def render_git_describe(pieces):
    "TAG[-DISTANCE-gHEX][-dirty].\n\n    Like 'git describe --tags --dirty --always'.\n\n    Exceptions:\n    1: no tags. HEX[-dirty]  (note: no 'g' prefix)\n    "
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance']:
            rendered += ('-%d-g%s' % (pieces['distance'], pieces['short']))
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render_git_describe_long(pieces):
    "TAG-DISTANCE-gHEX[-dirty].\n\n    Like 'git describe --tags --dirty --always -long'.\n    The distance/hash is unconditional.\n\n    Exceptions:\n    1: no tags. HEX[-dirty]  (note: no 'g' prefix)\n    "
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        rendered += ('-%d-g%s' % (pieces['distance'], pieces['short']))
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered

def render(pieces, style):
    'Render the given version pieces into the requested style.'
    if pieces['error']:
        return {'version': 'unknown', 'full-revisionid': pieces.get('long'), 'dirty': None, 'error': pieces['error'], 'date': None}
    if ((not style) or (style == 'default')):
        style = 'pep440'
    if (style == 'pep440'):
        rendered = render_pep440(pieces)
    elif (style == 'pep440-pre'):
        rendered = render_pep440_pre(pieces)
    elif (style == 'pep440-post'):
        rendered = render_pep440_post(pieces)
    elif (style == 'pep440-old'):
        rendered = render_pep440_old(pieces)
    elif (style == 'git-describe'):
        rendered = render_git_describe(pieces)
    elif (style == 'git-describe-long'):
        rendered = render_git_describe_long(pieces)
    else:
        raise ValueError(("unknown style '%s'" % style))
    return {'version': rendered, 'full-revisionid': pieces['long'], 'dirty': pieces['dirty'], 'error': None, 'date': pieces.get('date')}

class VersioneerBadRootError(Exception):
    'The project root directory is unknown or missing key files.'

def get_versions(verbose=False):
    "Get the project version from whatever source is available.\n\n    Returns dict with two keys: 'version' and 'full'.\n    "
    if ('versioneer' in sys.modules):
        del sys.modules['versioneer']
    root = get_root()
    cfg = get_config_from_root(root)
    assert (cfg.VCS is not None), 'please set [versioneer]VCS= in setup.cfg'
    handlers = HANDLERS.get(cfg.VCS)
    assert handlers, ("unrecognized VCS '%s'" % cfg.VCS)
    verbose = (verbose or cfg.verbose)
    assert (cfg.versionfile_source is not None), 'please set versioneer.versionfile_source'
    assert (cfg.tag_prefix is not None), 'please set versioneer.tag_prefix'
    versionfile_abs = os.path.join(root, cfg.versionfile_source)
    get_keywords_f = handlers.get('get_keywords')
    from_keywords_f = handlers.get('keywords')
    if (get_keywords_f and from_keywords_f):
        try:
            keywords = get_keywords_f(versionfile_abs)
            ver = from_keywords_f(keywords, cfg.tag_prefix, verbose)
            if verbose:
                print(('got version from expanded keyword %s' % ver))
            return ver
        except NotThisMethod:
            pass
    try:
        ver = versions_from_file(versionfile_abs)
        if verbose:
            print(f'got version from file {versionfile_abs} {ver}')
        return ver
    except NotThisMethod:
        pass
    from_vcs_f = handlers.get('pieces_from_vcs')
    if from_vcs_f:
        try:
            pieces = from_vcs_f(cfg.tag_prefix, root, verbose)
            ver = render(pieces, cfg.style)
            if verbose:
                print(('got version from VCS %s' % ver))
            return ver
        except NotThisMethod:
            pass
    try:
        if cfg.parentdir_prefix:
            ver = versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
            if verbose:
                print(('got version from parentdir %s' % ver))
            return ver
    except NotThisMethod:
        pass
    if verbose:
        print('unable to compute version')
    return {'version': '0+unknown', 'full-revisionid': None, 'dirty': None, 'error': 'unable to compute version', 'date': None}

def get_version():
    'Get the short version string for this project.'
    return get_versions()['version']

def get_cmdclass(cmdclass=None):
    'Get the custom setuptools/distutils subclasses used by Versioneer.\n\n    If the package uses a different cmdclass (e.g. one from numpy), it\n    should be provide as an argument.\n    '
    if ('versioneer' in sys.modules):
        del sys.modules['versioneer']
    cmds = ({} if (cmdclass is None) else cmdclass.copy())
    from distutils.core import Command

    class cmd_version(Command):
        description = 'report generated version string'
        user_options = []
        boolean_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            vers = get_versions(verbose=True)
            print(('Version: %s' % vers['version']))
            print((' full-revisionid: %s' % vers.get('full-revisionid')))
            print((' dirty: %s' % vers.get('dirty')))
            print((' date: %s' % vers.get('date')))
            if vers['error']:
                print((' error: %s' % vers['error']))
    cmds['version'] = cmd_version
    if ('build_py' in cmds):
        _build_py = cmds['build_py']
    elif ('setuptools' in sys.modules):
        from setuptools.command.build_py import build_py as _build_py
    else:
        from distutils.command.build_py import build_py as _build_py

    class cmd_build_py(_build_py):

        def run(self):
            root = get_root()
            cfg = get_config_from_root(root)
            versions = get_versions()
            _build_py.run(self)
            if cfg.versionfile_build:
                target_versionfile = os.path.join(self.build_lib, cfg.versionfile_build)
                print(('UPDATING %s' % target_versionfile))
                write_to_version_file(target_versionfile, versions)
    cmds['build_py'] = cmd_build_py
    if ('setuptools' in sys.modules):
        from setuptools.command.build_ext import build_ext as _build_ext
    else:
        from distutils.command.build_ext import build_ext as _build_ext

    class cmd_build_ext(_build_ext):

        def run(self):
            root = get_root()
            cfg = get_config_from_root(root)
            versions = get_versions()
            _build_ext.run(self)
            if self.inplace:
                return
            target_versionfile = os.path.join(self.build_lib, cfg.versionfile_source)
            print(('UPDATING %s' % target_versionfile))
            write_to_version_file(target_versionfile, versions)
    cmds['build_ext'] = cmd_build_ext
    if ('cx_Freeze' in sys.modules):
        from cx_Freeze.dist import build_exe as _build_exe

        class cmd_build_exe(_build_exe):

            def run(self):
                root = get_root()
                cfg = get_config_from_root(root)
                versions = get_versions()
                target_versionfile = cfg.versionfile_source
                print(('UPDATING %s' % target_versionfile))
                write_to_version_file(target_versionfile, versions)
                _build_exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, 'w') as f:
                    LONG = LONG_VERSION_PY[cfg.VCS]
                    f.write((LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source}))
        cmds['build_exe'] = cmd_build_exe
        del cmds['build_py']
    if ('py2exe' in sys.modules):
        from py2exe.distutils_buildexe import py2exe as _py2exe

        class cmd_py2exe(_py2exe):

            def run(self):
                root = get_root()
                cfg = get_config_from_root(root)
                versions = get_versions()
                target_versionfile = cfg.versionfile_source
                print(('UPDATING %s' % target_versionfile))
                write_to_version_file(target_versionfile, versions)
                _py2exe.run(self)
                os.unlink(target_versionfile)
                with open(cfg.versionfile_source, 'w') as f:
                    LONG = LONG_VERSION_PY[cfg.VCS]
                    f.write((LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source}))
        cmds['py2exe'] = cmd_py2exe
    if ('sdist' in cmds):
        _sdist = cmds['sdist']
    elif ('setuptools' in sys.modules):
        from setuptools.command.sdist import sdist as _sdist
    else:
        from distutils.command.sdist import sdist as _sdist

    class cmd_sdist(_sdist):

        def run(self):
            versions = get_versions()
            self._versioneer_generated_versions = versions
            self.distribution.metadata.version = versions['version']
            return _sdist.run(self)

        def make_release_tree(self, base_dir, files):
            root = get_root()
            cfg = get_config_from_root(root)
            _sdist.make_release_tree(self, base_dir, files)
            target_versionfile = os.path.join(base_dir, cfg.versionfile_source)
            print(('UPDATING %s' % target_versionfile))
            write_to_version_file(target_versionfile, self._versioneer_generated_versions)
    cmds['sdist'] = cmd_sdist
    return cmds
CONFIG_ERROR = "\nsetup.cfg is missing the necessary Versioneer configuration. You need\na section like:\n\n [versioneer]\n VCS = git\n style = pep440\n versionfile_source = src/myproject/_version.py\n versionfile_build = myproject/_version.py\n tag_prefix =\n parentdir_prefix = myproject-\n\nYou will also need to edit your setup.py to use the results:\n\n import versioneer\n setup(version=versioneer.get_version(),\n       cmdclass=versioneer.get_cmdclass(), ...)\n\nPlease read the docstring in ./versioneer.py for configuration instructions,\nedit setup.cfg, and re-run the installer or 'python versioneer.py setup'.\n"
SAMPLE_CONFIG = "\n# See the docstring in versioneer.py for instructions. Note that you must\n# re-run 'versioneer.py setup' after changing this section, and commit the\n# resulting files.\n\n[versioneer]\n#VCS = git\n#style = pep440\n#versionfile_source =\n#versionfile_build =\n#tag_prefix =\n#parentdir_prefix =\n\n"
INIT_PY_SNIPPET = "\nfrom ._version import get_versions\n__version__ = get_versions()['version']\ndel get_versions\n"

def do_setup():
    'Do main VCS-independent setup function for installing Versioneer.'
    root = get_root()
    try:
        cfg = get_config_from_root(root)
    except (OSError, configparser.NoSectionError, configparser.NoOptionError) as e:
        if isinstance(e, (EnvironmentError, configparser.NoSectionError)):
            print('Adding sample versioneer config to setup.cfg', file=sys.stderr)
            with open(os.path.join(root, 'setup.cfg'), 'a') as f:
                f.write(SAMPLE_CONFIG)
        print(CONFIG_ERROR, file=sys.stderr)
        return 1
    print((' creating %s' % cfg.versionfile_source))
    with open(cfg.versionfile_source, 'w') as f:
        LONG = LONG_VERSION_PY[cfg.VCS]
        f.write((LONG % {'DOLLAR': '$', 'STYLE': cfg.style, 'TAG_PREFIX': cfg.tag_prefix, 'PARENTDIR_PREFIX': cfg.parentdir_prefix, 'VERSIONFILE_SOURCE': cfg.versionfile_source}))
    ipy = os.path.join(os.path.dirname(cfg.versionfile_source), '__init__.py')
    if os.path.exists(ipy):
        try:
            with open(ipy) as f:
                old = f.read()
        except OSError:
            old = ''
        if (INIT_PY_SNIPPET not in old):
            print((' appending to %s' % ipy))
            with open(ipy, 'a') as f:
                f.write(INIT_PY_SNIPPET)
        else:
            print((' %s unmodified' % ipy))
    else:
        print((" %s doesn't exist, ok" % ipy))
        ipy = None
    manifest_in = os.path.join(root, 'MANIFEST.in')
    simple_includes = set()
    try:
        with open(manifest_in) as f:
            for line in f:
                if line.startswith('include '):
                    for include in line.split()[1:]:
                        simple_includes.add(include)
    except OSError:
        pass
    if ('versioneer.py' not in simple_includes):
        print(" appending 'versioneer.py' to MANIFEST.in")
        with open(manifest_in, 'a') as f:
            f.write('include versioneer.py\n')
    else:
        print(" 'versioneer.py' already in MANIFEST.in")
    if (cfg.versionfile_source not in simple_includes):
        print((" appending versionfile_source ('%s') to MANIFEST.in" % cfg.versionfile_source))
        with open(manifest_in, 'a') as f:
            f.write(('include %s\n' % cfg.versionfile_source))
    else:
        print(' versionfile_source already in MANIFEST.in')
    do_vcs_install(manifest_in, cfg.versionfile_source, ipy)
    return 0

def scan_setup_py():
    "Validate the contents of setup.py against Versioneer's expectations."
    found = set()
    setters = False
    errors = 0
    with open('setup.py') as f:
        for line in f.readlines():
            if ('import versioneer' in line):
                found.add('import')
            if ('versioneer.get_cmdclass()' in line):
                found.add('cmdclass')
            if ('versioneer.get_version()' in line):
                found.add('get_version')
            if ('versioneer.VCS' in line):
                setters = True
            if ('versioneer.versionfile_source' in line):
                setters = True
    if (len(found) != 3):
        print('')
        print('Your setup.py appears to be missing some important items')
        print('(but I might be wrong). Please make sure it has something')
        print('roughly like the following:')
        print('')
        print(' import versioneer')
        print(' setup( version=versioneer.get_version(),')
        print('        cmdclass=versioneer.get_cmdclass(),  ...)')
        print('')
        errors += 1
    if setters:
        print("You should remove lines like 'versioneer.VCS = ' and")
        print("'versioneer.versionfile_source = ' . This configuration")
        print('now lives in setup.cfg, and should be removed from setup.py')
        print('')
        errors += 1
    return errors
if (__name__ == '__main__'):
    cmd = sys.argv[1]
    if (cmd == 'setup'):
        errors = do_setup()
        errors += scan_setup_py()
        if errors:
            sys.exit(1)
