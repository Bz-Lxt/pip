from __future__ import annotations

import contextlib
import errno
import fnmatch
import os
import os.path
import random
import shutil
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, cast

from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size, hash_file
from pip._internal.utils.retry import retry


def check_path_owner(path: str) -> bool:
    # If we don't have a way to check the effective uid of this process, then
    # we'll just assume that we own the directory.
    if sys.platform == "win32" or not hasattr(os, "geteuid"):
        return True

    assert os.path.isabs(path)

    previous = None
    while path != previous:
        if os.path.lexists(path):
            # Check if path is writable by current user.
            if os.geteuid() == 0:
                # Special handling for root user in order to handle properly
                # cases where users use sudo without -H flag.
                try:
                    path_uid = get_path_uid(path)
                except OSError:
                    return False
                return path_uid == 0
            else:
                return os.access(path, os.W_OK)
        else:
            previous, path = path, os.path.dirname(path)
    return False  # assume we don't own the path


@contextmanager
def adjacent_tmp_file(path: str, **kwargs: Any) -> Generator[BinaryIO, None, None]:
    """Return a file-like object pointing to a tmp file next to path.

    The file is created securely and is ensured to be written to disk
    after the context reaches its end.

    kwargs will be passed to tempfile.NamedTemporaryFile to control
    the way the temporary file will be opened.
    """
    with NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path),
        prefix=os.path.basename(path),
        suffix=".tmp",
        **kwargs,
    ) as f:
        result = cast(BinaryIO, f)
        try:
            yield result
        finally:
            result.flush()
            os.fsync(result.fileno())


replace = retry(stop_after_delay=1, wait=0.25)(os.replace)


def files_are_identical(path_a: str, path_b: str) -> bool:
    """Return whether two paths point to files with byte-identical contents.

    A size comparison short-circuits hashing files that obviously differ.
    Missing or unreadable files compare as non-identical rather than
    raising, so callers can use this while racing with other processes.
    """
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        hash_a, _ = hash_file(path_a)
        hash_b, _ = hash_file(path_b)
    except OSError:
        return False
    return hash_a.digest() == hash_b.digest()


@dataclass(frozen=True)
class PublishedFile:
    """The result of publishing a file with :func:`publish_file`."""

    #: The path the published contents ended up at (``dest`` or an
    #: alternative collision-free name).
    path: str
    #: False when the path already existed with identical contents and was
    #: therefore reused as-is; True when new contents were written.
    created: bool
    #: When incoming contents won the canonical name, the path at which the
    #: conflicting pre-existing contents were preserved; otherwise None.
    displaced_to: str | None = None


#: Derive an alternative name for a file whose canonical destination is
#: already occupied by *different* contents. ``dest`` is the canonical
#: path, ``src`` the file being published, and ``index`` is zero for the
#: first alternative and incremented if that name is taken by yet another
#: distinct content. The result MUST be deterministic for given arguments
#: so that retries and concurrent publishers converge on the same layout.
AlternativeName = Callable[[str, str, int], str]

#: Errnos raised by ``os.link`` when the filesystem does not support hard
#: links. The name is created exclusively (never replacing anything), so a
#: cross-device link error must be worked around by staging a copy first.
_HARDLINK_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }
)


def _create_without_clobbering(src: str, dest: str) -> None:
    """Create ``dest`` with the contents of ``src``, atomically.

    ``dest`` is never replaced: if it already exists (including as a
    dangling symlink), :class:`FileExistsError` is raised. The preferred
    implementation is a hard link, which publishes the complete file in a
    single atomic operation. When the source lives on a different
    filesystem, or hard links are unsupported, the contents are staged in
    an adjacent temporary file and the link/rename is done from there.
    """
    try:
        os.link(src, dest)
        return
    except FileExistsError:
        raise
    except OSError as e:
        if e.errno == errno.EXDEV:
            pass
        elif e.errno not in _HARDLINK_UNSUPPORTED_ERRNOS:
            raise

    # Stage the contents next to the destination, which is on the same
    # filesystem, so the final hard link cannot cross devices again.
    with adjacent_tmp_file(dest) as tmp:
        with open(src, "rb") as src_file:
            shutil.copyfileobj(src_file, tmp)
        staged_path = tmp.name

    try:
        try:
            os.link(staged_path, dest)
            return
        except FileExistsError:
            raise
        except OSError as e:
            if e.errno not in _HARDLINK_UNSUPPORTED_ERRNOS:
                raise

        # Filesystems without hard link support (rare): claim the name
        # exclusively first, so nothing is ever replaced, then move the
        # staged contents into the claimed name.
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(fd)
        replace(staged_path, dest)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(staged_path)


def _publish_staged_file(
    staged: str,
    dest: str,
    alternative_name: AlternativeName | None,
) -> PublishedFile:
    """Publish the complete regular file ``staged`` at ``dest``.

    The existing occupant keeps the canonical name; diverging incoming
    contents are diverted. See :func:`publish_file` for the semantics.
    """
    target = dest
    index = 0
    while True:
        if os.path.lexists(target):
            # A same-named file is only reused when its contents are
            # byte-identical to what is being published.
            if files_are_identical(target, staged):
                return PublishedFile(target, created=False)
            if alternative_name is None:
                raise FileExistsError(target)
            # Different contents: the existing file stays where it is and
            # the new file is diverted to a deterministic alternative.
            target = alternative_name(dest, staged, index)
            index += 1
            if index > 100:
                raise OSError(
                    errno.EEXIST,
                    "Could not find a collision-free file name",
                    dest,
                )
            continue
        try:
            _create_without_clobbering(staged, target)
        except FileExistsError:
            # Another process published the name concurrently; re-check
            # whether it ended up with identical contents.
            continue
        return PublishedFile(target, created=True)


@contextmanager
def _stage_next_to(src: str, dest: str) -> Generator[str, None, None]:
    """Yield a path next to ``dest`` holding the complete contents of ``src``.

    A hard link is used when the filesystem supports it; otherwise the
    contents are copied. In both cases ``src`` is left in place, so the
    same source can be published concurrently by several processes.
    """
    with adjacent_tmp_file(dest) as tmp:
        staged_path = tmp.name
    # The temp file starts out empty; remove it so os.link can claim the
    # unique name instead.
    os.unlink(staged_path)
    linked = False
    try:
        os.link(src, staged_path)
        linked = True
    except FileExistsError:
        # The temp name is unique in practice; fall back to copying.
        pass
    except OSError as e:
        if e.errno != errno.EXDEV and e.errno not in _HARDLINK_UNSUPPORTED_ERRNOS:
            raise

    if not linked:
        with adjacent_tmp_file(dest) as tmp:
            staged_path = tmp.name
            with open(src, "rb") as src_file:
                shutil.copyfileobj(src_file, tmp)

    try:
        yield staged_path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(staged_path)


def _replace_with(src: str, dest: str) -> None:
    """Atomically replace ``dest`` with ``src``, staging across devices.

    ``src`` itself is never moved, and the final rename happens within
    ``dest``'s directory. The caller is responsible for having already
    preserved whatever ``dest`` previously held.
    """
    with _stage_next_to(src, dest) as staged:
        replace(staged, dest)


def _publish_incoming_wins(
    src: str,
    dest: str,
    alternative_name: AlternativeName | None,
) -> PublishedFile:
    """Publish ``src`` at ``dest``, preserving a displaced occupant.

    The incoming contents are authoritative (e.g. a download that passed
    hash verification): conflicting existing contents are first preserved
    under a deterministic name derived from *their own* contents, then the
    canonical name is atomically replaced. See :func:`publish_file`.
    """
    index = 0
    while True:
        if not os.path.lexists(dest):
            try:
                _create_without_clobbering(src, dest)
            except FileExistsError:
                # Another process claimed the name concurrently; compare.
                continue
            return PublishedFile(dest, created=True)

        if files_are_identical(src, dest):
            return PublishedFile(dest, created=False)

        if alternative_name is None:
            raise FileExistsError(dest)

        # A racing publisher may have already replaced the occupant with
        # the same contents we hold; re-check before backing anything up.
        if files_are_identical(src, dest):
            return PublishedFile(dest, created=False)

        # Preserve the existing contents under a name derived from those
        # contents (note: src argument is the existing file itself), so
        # retries and racing processes converge on the same backup.
        backup = alternative_name(dest, dest, index)
        try:
            preserved = publish_file(dest, backup)
        except FileExistsError:
            index += 1
            if index > 100:
                raise OSError(
                    errno.EEXIST,
                    "Could not find a collision-free file name",
                    dest,
                ) from None
            continue

        _replace_with(src, dest)
        return PublishedFile(dest, created=True, displaced_to=preserved.path)


def publish_file(
    src: str,
    dest: str,
    alternative_name: AlternativeName | None = None,
    *,
    incoming_wins: bool = False,
) -> PublishedFile:
    """Publish ``src`` at ``dest`` without ever destroying distinct contents.

    - If ``dest`` does not exist, it is created atomically with the exact
      contents of ``src``.
    - If ``dest`` already holds byte-identical contents, it is reused
      as-is: nothing is copied, replaced, or deleted.
    - If ``dest`` holds different contents, behavior depends on
      ``incoming_wins``:

      * ``False`` (default): the existing file is left untouched and the
        incoming file is published at the path returned by
        ``alternative_name`` instead. Use this when neither side is
        authoritative, e.g. wheels landing in a shared build cache.
      * ``True``: the existing file is first preserved under the path
        returned by ``alternative_name`` (derived from the *existing*
        file's contents), and ``dest`` is then atomically replaced with
        ``src``. Use this when the incoming contents are authoritative,
        e.g. a fresh download that passed hash verification.

    ``alternative_name`` must derive a deterministic name (conventionally
    from a hash of its ``src`` argument), so retries are idempotent. If it
    is ``None``, :class:`FileExistsError` is raised on a content conflict.

    Publication creates or replaces the destination file atomically and
    never destroys an existing one, so concurrent publishers converge
    deterministically: identical contents always end up sharing the same
    path, and distinct contents get distinct paths.

    The directory containing ``dest`` must already exist.
    """
    if incoming_wins:
        return _publish_incoming_wins(src, dest, alternative_name)
    return _publish_staged_file(src, dest, alternative_name)


def publish_data(
    dest: str,
    data: bytes,
    alternative_name: AlternativeName | None = None,
) -> PublishedFile:
    """Atomically write in-memory ``data`` to ``dest``.

    This is the in-memory counterpart of :func:`publish_file`, with the
    same identity, conflict-protection, and concurrency semantics.
    """
    with adjacent_tmp_file(dest) as tmp:
        tmp.write(data)
        staged_path = tmp.name
    try:
        return _publish_staged_file(staged_path, dest, alternative_name)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(staged_path)


# test_writable_dir and _test_writable_dir_win are copied from Flit,
# with the author's agreement to also place them under pip's license.
def test_writable_dir(path: str) -> bool:
    """Check if a directory is writable.

    Uses os.access() on POSIX, tries creating files on Windows.
    """
    # If the directory doesn't exist, find the closest parent that does.
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break  # Should never get here, but infinite loops are bad
        path = parent

    if os.name == "posix":
        return os.access(path, os.W_OK)

    return _test_writable_dir_win(path)


def _test_writable_dir_win(path: str) -> bool:
    # os.access doesn't work on Windows: http://bugs.python.org/issue2528
    # and we can't use tempfile: http://bugs.python.org/issue22107
    basename = "accesstest_deleteme_fishfingers_custard_"
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(10):
        name = basename + "".join(random.choice(alphabet) for _ in range(6))
        file = os.path.join(path, name)
        try:
            fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            pass
        except PermissionError:
            # This could be because there's a directory with the same name.
            # But it's highly unlikely there's a directory called that,
            # so we'll assume it's because the parent dir is not writable.
            # This could as well be because the parent dir is not readable,
            # due to non-privileged user access.
            return False
        else:
            os.close(fd)
            os.unlink(file)
            return True

    # This should never be reached
    raise OSError("Unexpected condition testing for writable directory")


def find_files(path: str, pattern: str) -> list[str]:
    """Returns a list of absolute paths of files beneath path, recursively,
    with filenames which match the UNIX-style shell glob pattern."""
    result: list[str] = []
    for root, _, files in os.walk(path):
        matches = fnmatch.filter(files, pattern)
        result.extend(os.path.join(root, f) for f in matches)
    return result


def file_size(path: str) -> int | float:
    # If it's a symlink, return 0.
    if os.path.islink(path):
        return 0
    return os.path.getsize(path)


def format_file_size(path: str) -> str:
    return format_size(file_size(path))


def directory_size(path: str) -> int | float:
    size = 0.0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            size += file_size(file_path)
    return size


def format_directory_size(path: str) -> str:
    return format_size(directory_size(path))


def copy_directory_permissions(directory: str, target_file: BinaryIO) -> None:
    mode = (
        os.stat(directory).st_mode & 0o666  # select read/write permissions of directory
        | 0o600  # set owner read/write permissions
    )
    # Change permissions only if there is no risk of following a symlink.
    if os.chmod in os.supports_fd:
        os.chmod(target_file.fileno(), mode)
    elif os.chmod in os.supports_follow_symlinks:
        os.chmod(target_file.name, mode, follow_symlinks=False)


def _subdirs_without_generic(
    path: str, predicate: Callable[[str, list[str]], bool]
) -> Generator[Path]:
    """Yields every subdirectory of +path+ that has no files matching the
    predicate under it."""

    directories = []
    excluded: set[Path] = set()

    for root_str, _, filenames in os.walk(Path(path).resolve()):
        root = Path(root_str)
        if predicate(root_str, filenames):
            # This directory should be excluded, so exclude it and all of its
            # parent directories.
            # The last item in root.parents is ".", so we ignore it.
            excluded.update(root.parents[:-1])
            excluded.add(root)
        directories.append(root)

    for d in sorted(directories, reverse=True):
        if d not in excluded:
            yield d


def subdirs_without_files(path: str) -> Generator[Path]:
    """Yields every subdirectory of +path+ that has no files under it."""
    return _subdirs_without_generic(path, lambda root, filenames: len(filenames) > 0)


def subdirs_without_wheels(path: str) -> Generator[Path]:
    """Yields every subdirectory of +path+ that has no .whl files under it."""
    return _subdirs_without_generic(
        path, lambda root, filenames: any(x.endswith(".whl") for x in filenames)
    )
