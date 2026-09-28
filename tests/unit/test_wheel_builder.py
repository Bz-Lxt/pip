from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from pip._internal import wheel_builder
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.git import Git

from tests.lib import _create_test_package


@pytest.mark.parametrize(
    "s, expected",
    [
        # Trivial.
        ("pip-18.0", True),
        # Ambiguous.
        ("foo-2-2", True),
        ("im-valid", True),
        # Invalid.
        ("invalid", False),
        ("im_invalid", False),
    ],
)
def test_contains_egg_info(s: str, expected: bool) -> None:
    result = wheel_builder._contains_egg_info(s)
    assert result == expected


@dataclass
class ReqMock:
    name: str = "pendulum"
    is_wheel: bool = False
    editable: bool = False
    link: Link | None = None
    constraint: bool = False
    source_dir: str | None = "/tmp/pip-install-123/pendulum"
    supports_pyproject_editable: bool = False


@pytest.mark.parametrize(
    "req, expected",
    [
        (ReqMock(editable=True), False),
        (ReqMock(source_dir=None), False),
        (ReqMock(link=Link("git+https://g.c/org/repo")), False),
        (ReqMock(link=Link("https://g.c/dist.tgz")), False),
        (ReqMock(link=Link("https://g.c/dist-2.0.4.tgz")), True),
    ],
)
def test_should_cache(req: ReqMock, expected: bool) -> None:
    assert wheel_builder._should_cache(cast(InstallRequirement, req)) is expected


def test_should_cache_git_sha(tmpdir: Path) -> None:
    repo_path = os.fspath(_create_test_package(tmpdir, name="mypkg"))
    commit = Git.get_revision(repo_path)

    # a link referencing a sha should be cached
    url = "git+https://g.c/o/r@" + commit + "#egg=mypkg"
    req = ReqMock(link=Link(url), source_dir=repo_path)
    assert wheel_builder._should_cache(cast(InstallRequirement, req))

    # a link not referencing a sha should not be cached
    url = "git+https://g.c/o/r@master#egg=mypkg"
    req = ReqMock(link=Link(url), source_dir=repo_path)
    assert not wheel_builder._should_cache(cast(InstallRequirement, req))


@pytest.mark.parametrize("name", ["package", "my-package"])
def test_should_not_cache_local_directory(tmpdir: Path, name: str) -> None:
    project_dir = tmpdir / name
    project_dir.mkdir()
    req = ReqMock(link=Link(path_to_url(str(project_dir))))
    assert not wheel_builder._should_cache(cast(InstallRequirement, req))


def test_alternative_wheel_path_is_deterministic_and_valid(tmpdir: Path) -> None:
    import hashlib

    from pip._internal.models.wheel import Wheel

    payload = b"wheel contents"
    src = tmpdir / "built.whl"
    src.write_bytes(payload)
    dest = str(tmpdir / "package-1.0-py3-none-any.whl")

    token = hashlib.sha256(payload).hexdigest()[:12]
    expected = f"package-1.0-1{token}-py3-none-any.whl"
    expected_indexed = f"package-1.0-1{token}.1-py3-none-any.whl"

    first = wheel_builder._alternative_wheel_path(dest, str(src), 0)
    second = wheel_builder._alternative_wheel_path(dest, str(src), 0)
    indexed = wheel_builder._alternative_wheel_path(dest, str(src), 1)

    assert first == second
    assert os.path.basename(first) == expected
    assert os.path.basename(indexed) == expected_indexed
    # Both names stay valid PEP 427 wheel file names.
    for name in (first, indexed):
        wheel = Wheel(os.path.basename(name))
        assert wheel.name == "package"
        assert wheel.version == "1.0"
        assert wheel.build_tag is not None
