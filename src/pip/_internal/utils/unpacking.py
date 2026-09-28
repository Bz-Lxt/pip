"""Utilities related archives."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
import tarfile
import zipfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from zipfile import ZipInfo

from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
    BZ2_EXTENSIONS,
    TAR_EXTENSIONS,
    XZ_EXTENSIONS,
    ZIP_EXTENSIONS,
)
from pip._internal.utils.misc import ensure_dir

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = ZIP_EXTENSIONS + TAR_EXTENSIONS

try:
    import bz2  # noqa

    SUPPORTED_EXTENSIONS += BZ2_EXTENSIONS
except ImportError:
    logger.debug("bz2 module is not available")

try:
    # Only for Python 3.3+
    import lzma  # noqa

    SUPPORTED_EXTENSIONS += XZ_EXTENSIONS
except ImportError:
    logger.debug("lzma module is not available")


class ExtractionLimitsExceeded(InstallationError):
    """Raised when a command would unpack more than its configured limits.

    Source distributions and wheels can expand into a file set many times
    larger than the compressed download. Raising this error aborts the
    command *before* the offending member is written, so callers can clean
    up the partially extracted files, skip publishing anything to the wheel
    cache, and leave the target environment untouched.
    """

    def __init__(
        self,
        *,
        kind: str,
        limit: int,
        used: int,
        archive: str,
        member: str | None = None,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.used = used
        self.archive = archive
        self.member = member

        if member is not None:
            subject = f"member {member!r} of {archive!r}"
        else:
            subject = f"{archive!r}"
        message = (
            f"The unpacked content {kind} limit of {limit} set for this "
            f"command would be exceeded while extracting {subject}: "
            f"{used} {kind} would have been unpacked in total. "
            "Aborting; partially extracted files have been removed, "
            "nothing was installed and the wheel cache was not modified."
        )
        super().__init__(message)


@dataclass
class _ExtractionBudget:
    """Command-wide running total of content produced by unpacking.

    Every extracted archive member is charged *before* it is written to
    disk, using the uncompressed size advertised by the archive metadata,
    so a single oversized member cannot land on disk first.
    """

    max_bytes: int | None
    max_files: int | None
    used_bytes: int = 0
    used_files: int = 0
    # Real paths of wheels whose expansion has already been accounted for,
    # so a wheel is never counted twice (once when published to the cache
    # and once when installed into the environment).
    _reserved_wheels: set[str] = field(default_factory=set)

    def charge(
        self,
        num_bytes: int = 0,
        num_files: int = 0,
        *,
        archive: str,
        member: str | None = None,
    ) -> None:
        """Account for ``num_bytes``/``num_files`` before they are written."""
        would_be_bytes = self.used_bytes + num_bytes
        would_be_files = self.used_files + num_files

        if self.max_bytes is not None and would_be_bytes > self.max_bytes:
            raise ExtractionLimitsExceeded(
                kind="bytes",
                limit=self.max_bytes,
                used=would_be_bytes,
                archive=archive,
                member=member,
            )
        if self.max_files is not None and would_be_files > self.max_files:
            raise ExtractionLimitsExceeded(
                kind="files",
                limit=self.max_files,
                used=would_be_files,
                archive=archive,
                member=member,
            )

        self.used_bytes = would_be_bytes
        self.used_files = would_be_files

    def reserve_wheel(self, wheel_path: str, *, key_path: str | None = None) -> None:
        """Charge for a wheel's expanded content without extracting it.

        The totals are read from the wheel's zip metadata, which is the same
        metadata used while installing. ``key_path`` allows accounting a wheel
        under the path it will have *after* it is published to the cache, so a
        later reservation for that published path is a no-op.
        """
        key = os.path.realpath(key_path or wheel_path)
        if key in self._reserved_wheels:
            return

        total_bytes, total_files = _wheel_extracted_size(wheel_path)
        self.charge(
            total_bytes,
            total_files,
            archive=os.path.basename(wheel_path),
        )
        self._reserved_wheels.add(key)


def _wheel_extracted_size(wheel_path: str) -> tuple[int, int]:
    """Return ``(uncompressed bytes, file count)`` for a wheel archive."""
    try:
        with zipfile.ZipFile(wheel_path, allowZip64=True) as wheel_zip:
            infos = wheel_zip.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise InstallationError(f"Cannot read wheel {wheel_path!r}: {exc}")

    total_bytes = 0
    total_files = 0
    for info in infos:
        if info.is_dir():
            continue
        total_bytes += info.file_size
        total_files += 1
    return total_bytes, total_files


_extraction_budget: _ExtractionBudget | None = None


@contextmanager
def extraction_limits(
    max_bytes: int | None,
    max_files: int | None,
) -> Generator[None, None, None]:
    """Install a command-wide budget on unpacked bytes and file count.

    The budget applies to every archive extracted while the context is
    active (sdists and wheels across all requirements), and is removed when
    the context exits. With both limits unset, extraction is unrestricted.
    """
    global _extraction_budget

    if max_bytes is None and max_files is None:
        yield
        return

    previous_budget = _extraction_budget
    _extraction_budget = _ExtractionBudget(max_bytes=max_bytes, max_files=max_files)
    try:
        yield
    finally:
        _extraction_budget = previous_budget


def charge_extraction(
    num_bytes: int = 0,
    num_files: int = 0,
    *,
    archive: str,
    member: str | None = None,
) -> None:
    """Charge an imminent extraction against the command-wide budget, if any."""
    if _extraction_budget is not None:
        _extraction_budget.charge(
            num_bytes,
            num_files,
            archive=archive,
            member=member,
        )


def reserve_wheel_extraction(
    wheel_path: str, *, key_path: str | None = None
) -> None:
    """Account for a wheel's expansion against the command-wide budget."""
    if _extraction_budget is not None:
        _extraction_budget.reserve_wheel(wheel_path, key_path=key_path)


def current_extraction_budget() -> _ExtractionBudget | None:
    return _extraction_budget


def current_umask() -> int:
    """Get the current umask which involves having to set it temporarily."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def split_leading_dir(path: str) -> list[str]:
    path = path.lstrip("/").lstrip("\\")
    if "/" in path and (
        ("\\" in path and path.find("/") < path.find("\\")) or "\\" not in path
    ):
        return path.split("/", 1)
    elif "\\" in path:
        return path.split("\\", 1)
    else:
        return [path, ""]


def has_leading_dir(paths: Iterable[str]) -> bool:
    """Returns true if all the paths have the same leading path name
    (i.e., everything is in one subdirectory in an archive)"""
    common_prefix = None
    for path in paths:
        prefix, rest = split_leading_dir(path)
        if not prefix:
            return False
        elif common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            return False
    return True


def is_within_directory(
    directory: str, target: str, *, resolve_symlinks: bool = False
) -> bool:
    """
    Return true if the absolute path of target is within the directory
    (including when target is equal to the directory).

    When ``resolve_symlinks`` is true, resolve symlinks before comparing so
    traversal through a symlink (e.g. "link/../file") is also caught.
    """
    if resolve_symlinks:
        abs_directory = os.path.realpath(directory)
        abs_target = os.path.realpath(target)
    else:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)

    return abs_target == abs_directory or abs_target.startswith(abs_directory + os.sep)


def _get_default_mode_plus_executable() -> int:
    return 0o777 & ~current_umask() | 0o111


def set_extracted_file_to_default_mode_plus_executable(path: str) -> None:
    """
    Make file present at path have execute for user/group/world
    (chmod +x) is no-op on windows per python docs
    """
    os.chmod(path, _get_default_mode_plus_executable())


def zip_item_is_executable(info: ZipInfo) -> bool:
    mode = info.external_attr >> 16
    # if mode and regular file and any execute permissions for
    # user/group/world?
    return bool(mode and stat.S_ISREG(mode) and mode & 0o111)


def unzip_file(filename: str, location: str, flatten: bool = True) -> None:
    """
    Unzip the file (with path `filename`) to the destination `location`.  All
    files are written based on system defaults and umask (i.e. permissions are
    not preserved), except that regular file members with any execute
    permissions (user, group, or world) have "chmod +x" applied after being
    written. Note that for windows, any execute changes using os.chmod are
    no-ops per the python docs.
    """
    ensure_dir(location)
    zipfp = open(filename, "rb")
    try:
        zip = zipfile.ZipFile(zipfp, allowZip64=True)
        leading = has_leading_dir(zip.namelist()) and flatten
        for info in zip.infolist():
            name = info.filename
            fn = name
            if leading:
                fn = split_leading_dir(name)[1]
            fn = os.path.join(location, fn)
            dir = os.path.dirname(fn)
            if not is_within_directory(location, fn):
                message = (
                    "The zip file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise InstallationError(message.format(filename, fn, location))
            # Charge before creating anything, using the uncompressed size
            # advertised in the zip metadata. Directory entries count towards
            # the file (inode) limit as well.
            if fn.endswith(("/", "\\")):
                charge_extraction(0, 1, archive=filename, member=name)
                # A directory
                ensure_dir(fn)
            else:
                charge_extraction(
                    info.file_size, 1, archive=filename, member=name
                )
                ensure_dir(dir)
                # Don't use read() to avoid allocating an arbitrarily large
                # chunk of memory for the file's content
                fp = zip.open(name)
                try:
                    with open(fn, "wb") as destfp:
                        shutil.copyfileobj(fp, destfp)
                finally:
                    fp.close()
                    if zip_item_is_executable(info):
                        set_extracted_file_to_default_mode_plus_executable(fn)
    finally:
        zipfp.close()


def untar_file(filename: str, location: str) -> None:
    """
    Untar the file (with path `filename`) to the destination `location`.
    All files are written based on system defaults and umask (i.e. permissions
    are not preserved), except that regular file members with any execute
    permissions (user, group, or world) have "chmod +x" applied on top of the
    default.  Note that for windows, any execute changes using os.chmod are
    no-ops per the python docs.
    """
    ensure_dir(location)
    if filename.lower().endswith(".gz") or filename.lower().endswith(".tgz"):
        mode = "r:gz"
    elif filename.lower().endswith(BZ2_EXTENSIONS):
        mode = "r:bz2"
    elif filename.lower().endswith(XZ_EXTENSIONS):
        mode = "r:xz"
    elif filename.lower().endswith(".tar"):
        mode = "r"
    else:
        logger.warning(
            "Cannot determine compression type for file %s",
            filename,
        )
        mode = "r:*"

    tar = tarfile.open(filename, mode, encoding="utf-8")  # type: ignore
    try:
        leading = has_leading_dir([member.name for member in tar.getmembers()])

        # PEP 706 added `tarfile.data_filter`, and made some other changes to
        # Python's tarfile module (see below). The features were backported to
        # security releases.
        try:
            data_filter = tarfile.data_filter
        except AttributeError:
            _untar_without_filter(filename, location, tar, leading)
        else:
            default_mode_plus_executable = _get_default_mode_plus_executable()

            if leading:
                # Strip the leading directory from all files in the archive,
                # including hardlink targets (which are relative to the
                # unpack location).
                for member in tar.getmembers():
                    name_lead, name_rest = split_leading_dir(member.name)
                    member.name = name_rest
                    if member.islnk():
                        lnk_lead, lnk_rest = split_leading_dir(member.linkname)
                        if lnk_lead == name_lead:
                            member.linkname = lnk_rest

            def pip_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
                orig_mode = member.mode
                try:
                    try:
                        member = data_filter(member, location)
                    except tarfile.LinkOutsideDestinationError:
                        if sys.version_info[:3] in {
                            (3, 9, 17),
                            (3, 10, 12),
                            (3, 11, 4),
                        }:
                            # The tarfile filter in specific Python versions
                            # raises LinkOutsideDestinationError on valid input
                            # (https://github.com/python/cpython/issues/107845)
                            # Ignore the error there, but do use the
                            # more lax `tar_filter`
                            member = tarfile.tar_filter(member, location)
                        else:
                            raise
                except tarfile.TarError as exc:
                    message = "Invalid member in the tar file {}: {}"
                    # Filter error messages mention the member name.
                    # No need to add it here.
                    raise InstallationError(
                        message.format(
                            filename,
                            exc,
                        )
                    )
                if member.isfile() and orig_mode & 0o111:
                    member.mode = default_mode_plus_executable
                else:
                    # See PEP 706 note above.
                    # The PEP changed this from `int` to `Optional[int]`,
                    # where None means "use the default". Mypy doesn't
                    # know this yet.
                    member.mode = None  # type: ignore [assignment]
                return member

            # Extract member by member instead of using extractall() so each
            # member can be charged against the command-wide extraction
            # budget *before* it is written. The security filter has already
            # been applied by pip_filter, so no filter is applied again.
            for member in tar.getmembers():
                member = pip_filter(member, location)
                # Every member creates a filesystem entry (directories and
                # links consume an inode too); only regular files have size.
                charge_extraction(
                    member.size, 1, archive=filename, member=member.name
                )
                tar.extract(member, location, filter="fully_trusted")

    finally:
        tar.close()


def is_symlink_target_in_tar(tar: tarfile.TarFile, tarinfo: tarfile.TarInfo) -> bool:
    """Check if the file pointed to by the symbolic link is in the tar archive"""
    linkname = os.path.join(os.path.dirname(tarinfo.name), tarinfo.linkname)

    linkname = os.path.normpath(linkname)
    linkname = linkname.replace("\\", "/")

    try:
        tar.getmember(linkname)
        return True
    except KeyError:
        return False


def _untar_without_filter(
    filename: str,
    location: str,
    tar: tarfile.TarFile,
    leading: bool,
) -> None:
    """Fallback for Python without tarfile.data_filter"""
    # NOTE: This function can be removed once pip requires CPython ≥ 3.12.​
    # PEP 706 added tarfile.data_filter, made tarfile extraction operations more secure.
    # This feature is fully supported from CPython 3.12 onward.
    for member in tar.getmembers():
        fn = member.name
        if leading:
            fn = split_leading_dir(fn)[1]
        path = os.path.join(location, fn)

        # Charge before creating the member, mirroring the data_filter path.
        charge_extraction(
            member.size, 1, archive=filename, member=member.name
        )

        # The plain check rejects textual ".." escapes; resolving symlinks also
        # catches a later member redirected outside by an earlier member's
        # symlink (e.g. "link/../file").
        if not is_within_directory(location, path) or not is_within_directory(
            location, path, resolve_symlinks=True
        ):
            message = (
                "The tar file ({}) has a file ({}) trying to install "
                "outside target directory ({})"
            )
            raise InstallationError(message.format(filename, path, location))
        if member.isdir():
            ensure_dir(path)
        elif member.issym():
            # Reject symlinks resolving outside the destination, so a later
            # member cannot be written through them.
            target = os.path.join(os.path.dirname(path), member.linkname)
            if not is_within_directory(location, target, resolve_symlinks=True):
                message = (
                    "The tar file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise InstallationError(
                    message.format(filename, member.name, member.linkname)
                )
            if not is_symlink_target_in_tar(tar, member):
                message = (
                    "The tar file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise InstallationError(
                    message.format(filename, member.name, member.linkname)
                )
            try:
                tar._extract_member(member, path)
            except Exception as exc:
                # Some corrupt tar files seem to produce this
                # (specifically bad symlinks)
                logger.warning(
                    "In the tar file %s the member %s is invalid: %s",
                    filename,
                    member.name,
                    exc,
                )
                continue
        else:
            try:
                fp = tar.extractfile(member)
            except (KeyError, AttributeError) as exc:
                # Some corrupt tar files seem to produce this
                # (specifically bad symlinks)
                logger.warning(
                    "In the tar file %s the member %s is invalid: %s",
                    filename,
                    member.name,
                    exc,
                )
                continue
            ensure_dir(os.path.dirname(path))
            assert fp is not None
            with open(path, "wb") as destfp:
                shutil.copyfileobj(fp, destfp)
            fp.close()
            # Update the timestamp (useful for cython compiled files)
            tar.utime(member, path)
            # member have any execute permissions for user/group/world?
            if member.mode & 0o111:
                set_extracted_file_to_default_mode_plus_executable(path)


def unpack_file(
    filename: str,
    location: str,
    content_type: str | None = None,
) -> None:
    """Unpack ``filename`` into ``location``.

    Archive format is chosen in order of decreasing reliability:
    ``content_type``, then filename extension, then magic signature
    (unambiguous matches only).
    """
    filename = os.path.realpath(filename)
    zip_flatten = not filename.endswith(".whl")

    def _unzip() -> None:
        unzip_file(filename, location, flatten=zip_flatten)

    def _untar() -> None:
        untar_file(filename, location)

    if content_type == "application/zip":
        do_unpack = _unzip
    elif content_type == "application/x-gzip":
        do_unpack = _untar
    elif filename.lower().endswith(ZIP_EXTENSIONS):
        do_unpack = _unzip
    elif filename.lower().endswith(TAR_EXTENSIONS + BZ2_EXTENSIONS + XZ_EXTENSIONS):
        do_unpack = _untar
    else:
        # avoid ambiguous case where both signature checks return True
        is_zipfile = zipfile.is_zipfile(filename)
        is_tarfile = tarfile.is_tarfile(filename)
        if is_zipfile and not is_tarfile:
            do_unpack = _unzip
        elif is_tarfile and not is_zipfile:
            do_unpack = _untar
        else:
            if is_zipfile and is_tarfile:
                logger.error("Ambiguous file signature in %s.", filename)

            logger.critical(
                "Cannot unpack file %s (downloaded from %s, content-type: %s); "
                "cannot detect archive format",
                filename,
                location,
                content_type,
            )
            raise InstallationError(f"Cannot determine archive format of {location}")

    try:
        do_unpack()
    except ExtractionLimitsExceeded:
        # The unpack was aborted mid-way: remove the partially extracted
        # tree right away so it cannot exhaust disk space or inodes, even
        # when --no-clean would otherwise keep pip's temporary directories.
        shutil.rmtree(location, ignore_errors=True)
        raise
