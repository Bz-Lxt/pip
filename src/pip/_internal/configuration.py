"""Configuration management setup

Some terminology:
- name
  As written in config files.
- value
  Value associated with a name
- key
  Name combined with it's section (section.name)
- variant
  A single word describing where the configuration key-value pair came from
"""

from __future__ import annotations

import configparser
import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NewType

from pip._internal.exceptions import (
    ConfigurationError,
    ConfigurationFileCouldNotBeLoaded,
)
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS, get_locale_encoding
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum

RawConfigParser = configparser.RawConfigParser  # Shorthand
Kind = NewType("Kind", str)

CONFIG_BASENAME = "pip.ini" if WINDOWS else "pip.conf"
ENV_NAMES_IGNORED = "version", "help"

# The kinds of configurations there are.
kinds = enum(
    USER="user",  # User Specific
    GLOBAL="global",  # System Wide
    SITE="site",  # [Virtual] Environment Specific
    ENV="env",  # from PIP_CONFIG_FILE
    ENV_VAR="env-var",  # from Environment Variables
)
OVERRIDE_ORDER = kinds.GLOBAL, kinds.USER, kinds.SITE, kinds.ENV, kinds.ENV_VAR
VALID_LOAD_ONLY = kinds.USER, kinds.GLOBAL, kinds.SITE

logger = getLogger(__name__)


# NOTE: Maybe use the optionx attribute to normalize keynames.
def _normalize_name(name: str) -> str:
    """Make a name consistent regardless of source (environment or file)"""
    name = name.lower().replace("_", "-")
    name = name.removeprefix("--")  # only prefer long opts
    return name


def _disassemble_key(name: str) -> list[str]:
    if "." not in name:
        error_message = (
            "Key does not contain dot separated section and key. "
            f"Perhaps you wanted to use 'global.{name}' instead?"
        )
        raise ConfigurationError(error_message)
    return name.split(".", 1)


def get_configuration_files() -> dict[Kind, list[str]]:
    global_config_files = [
        os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
    ]

    site_config_file = os.path.join(sys.prefix, CONFIG_BASENAME)
    legacy_config_file = os.path.join(
        os.path.expanduser("~"),
        "pip" if WINDOWS else ".pip",
        CONFIG_BASENAME,
    )
    new_config_file = os.path.join(appdirs.user_config_dir("pip"), CONFIG_BASENAME)
    return {
        kinds.GLOBAL: global_config_files,
        kinds.SITE: [site_config_file],
        kinds.USER: [legacy_config_file, new_config_file],
    }


@dataclass
class _Edit:
    """A single set/unset operation requested against a configuration file."""

    action: str  # One of "set", "unset"
    section: str
    name: str
    value: str = ""


class _InlineConfigEditor:
    """Apply set/unset edits to INI text without touching unrelated bytes.

    Unlike re-serializing the whole file with ``RawConfigParser.write``
    (which drops comments and reformats everything), this editor rewrites
    only the line(s) of the targeted option. Comments, blank lines,
    indentation, the choice of separator and line endings outside the
    target are preserved byte for byte.
    """

    # Mirror configparser's SECTCRE/OPTCRE; the section regex is matched
    # against the whitespace-stripped line, just like configparser does.
    _SECTION_RE = re.compile(r"\[(?P<name>[^]]+)\]")
    _OPTION_RE = re.compile(
        r"^(?P<indent>[ \t]*)"
        r"(?P<name>[^=:]+?)"
        r"(?P<lead>[ \t]*)(?P<vi>[=:])(?P<trail>[ \t]*)"
        r"(?P<value>.*)$"
    )
    _LINE_RE = re.compile(r".*?(?:\r\n|\r|\n)|.+$", re.DOTALL)

    def __init__(self, text: str) -> None:
        self._lines: list[str] = self._LINE_RE.findall(text) if text else []

    #
    # Line classification
    #

    @staticmethod
    def _is_comment(line: str) -> bool:
        return line.strip().startswith(("#", ";"))

    @staticmethod
    def _is_blank(line: str) -> bool:
        return line.strip() == ""

    @staticmethod
    def _indent_column(line: str) -> int:
        return len(line) - len(line.lstrip(" \t"))

    @staticmethod
    def _eol_of(line: str) -> str:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith(("\n", "\r")):
            return line[-1]
        return ""

    def _find_section(self, name: str) -> tuple[int, int] | None:
        """Return (header index, end index) of section ``name``."""
        header_index: int | None = None
        for index, line in enumerate(self._lines):
            match = self._SECTION_RE.match(line.strip())
            if match:
                if header_index is None and match.group("name") == name:
                    header_index = index
                elif header_index is not None:
                    return header_index, index
        if header_index is None:
            return None
        return header_index, len(self._lines)

    def _section_ranges(self) -> list[tuple[int, int]]:
        """(content start, end) pairs for every section in the file."""
        headers = [
            index
            for index, line in enumerate(self._lines)
            if self._SECTION_RE.match(line.strip())
        ]
        ranges = []
        for position, header in enumerate(headers):
            end = (
                headers[position + 1]
                if position + 1 < len(headers)
                else len(self._lines)
            )
            ranges.append((header + 1, end))
        return ranges

    def _iter_option_headers(
        self, start: int, end: int
    ) -> Iterable[tuple[int, int, re.Match[str]]]:
        """Yield (index, indent column, match) for each real option.

        Indented continuation lines belonging to a previous option are
        skipped, following configparser's indentation rules.
        """
        indent_level = -1
        for index in range(start, end):
            line = self._lines[index]
            if self._is_blank(line) or self._is_comment(line):
                continue
            column = self._indent_column(line)
            if indent_level >= 0 and column > indent_level:
                # A continuation line of the previous option.
                continue
            match = self._OPTION_RE.match(line)
            if match is None:
                # A bogus line resets continuation tracking, as in
                # configparser's own scanner.
                indent_level = column
                continue
            indent_level = column
            yield index, column, match

    def _find_option(self, start: int, end: int, name: str) -> int | None:
        for index, _, match in self._iter_option_headers(start, end):
            if _normalize_name(match.group("name").strip()) == name:
                return index
        return None

    def _value_span_end(self, index: int, indent: int) -> int:
        """Exclusive end of continuation lines after the option line."""
        end_index = index + 1
        while end_index < len(self._lines):
            line = self._lines[end_index]
            if self._is_blank(line) or self._is_comment(line):
                break
            if self._indent_column(line) > indent:
                end_index += 1
            else:
                break
        return end_index

    #
    # Style used for newly inserted lines
    #

    def _preferred_eol(self) -> str:
        for line in self._lines:
            eol = self._eol_of(line)
            if eol:
                return eol
        return os.linesep

    def _preferred_style(self) -> tuple[str, str, str, str]:
        """(indent, lead, separator, trail) of the first option, if any."""
        for start, end in self._section_ranges():
            for _, _, match in self._iter_option_headers(start, end):
                return (
                    match.group("indent"),
                    match.group("lead"),
                    match.group("vi"),
                    match.group("trail"),
                )
        return "", " ", "=", " "

    def _format_option_line(self, name: str, value: str) -> str:
        indent, lead, separator, trail = self._preferred_style()
        return f"{indent}{name}{lead}{separator}{trail}{value}{self._preferred_eol()}"

    #
    # Editing operations
    #

    def set_value(self, section: str, name: str, value: str) -> None:
        found = self._find_section(section)
        if found is None:
            self._append_section(section, name, value)
            return

        header, section_end = found
        option_index = self._find_option(header + 1, section_end, name)
        if option_index is None:
            self._insert_option(section_end, name, value)
            return

        match = self._OPTION_RE.match(self._lines[option_index])
        assert match is not None
        indent = self._indent_column(self._lines[option_index])
        span_end = self._value_span_end(option_index, indent)

        # Rewrite only this line, keeping key spelling, spacing and EOL.
        self._lines[option_index] = (
            match.group("indent")
            + match.group("name")
            + match.group("lead")
            + match.group("vi")
            + match.group("trail")
            + value
            + self._eol_of(self._lines[option_index])
        )
        # Old continuation lines are part of the replaced value and
        # would otherwise be appended to the new one.
        del self._lines[option_index + 1 : span_end]

    def unset_value(self, section: str, name: str) -> None:
        found = self._find_section(section)
        if found is None:
            raise ConfigurationError(f"No section {section!r} in configuration file")

        header, section_end = found
        option_index = self._find_option(header + 1, section_end, name)
        if option_index is None:
            raise ConfigurationError(
                f"No option {name!r} in section {section!r} of configuration file"
            )

        indent = self._indent_column(self._lines[option_index])
        span_end = self._value_span_end(option_index, indent)
        del self._lines[option_index:span_end]

        # Remove the section header when no options remain, matching the
        # historical behaviour. Comments and blank lines are left in place.
        found = self._find_section(section)
        assert found is not None
        header, section_end = found
        has_options = any(
            True for _ in self._iter_option_headers(header + 1, section_end)
        )
        if not has_options:
            del self._lines[header]

    def render(self) -> str:
        return "".join(self._lines)

    def _insert_option(self, section_end: int, name: str, value: str) -> None:
        line = self._format_option_line(name, value)
        # Place the new option after existing section content, ahead of
        # any trailing blank lines (and the next section header).
        index = section_end
        while index > 0 and self._is_blank(self._lines[index - 1]):
            index -= 1
        if index >= len(self._lines):
            eol = self._preferred_eol()
            if self._lines and self._eol_of(self._lines[-1]) == "":
                self._lines[-1] += eol
            self._lines.append(line)
        else:
            self._lines.insert(index, line)

    def _append_section(self, section: str, name: str, value: str) -> None:
        eol = self._preferred_eol()
        if self._lines:
            if self._eol_of(self._lines[-1]) == "":
                self._lines[-1] += eol
            if not self._is_blank(self._lines[-1]):
                self._lines.append(eol)
        self._lines.append(f"[{section}]{eol}")
        self._lines.append(self._format_option_line(name, value))


def _patch_config_text(text: str, edits: Iterable[_Edit]) -> str:
    """Apply ``edits`` to INI ``text``, returning the updated text."""
    editor = _InlineConfigEditor(text)
    for edit in edits:
        if edit.action == "set":
            editor.set_value(edit.section, edit.name, edit.value)
        elif edit.action == "unset":
            editor.unset_value(edit.section, edit.name)
        else:  # pragma: no cover - defensive guard
            raise ConfigurationError(f"Unknown configuration edit: {edit.action}")
    return editor.render()


class _ExclusiveFileLock:
    """Inter-process exclusive lock guarding configuration writes.

    An advisory byte lock is taken on a sibling ``<config-file>.lock``
    and held while the target is verified and atomically replaced,
    closing the verify-then-replace race window between cooperating
    pip processes. Writers that do not take the lock are still caught
    by the on-disk content comparison.
    """

    def __init__(self, fname: str) -> None:
        self._lock_path = f"{fname}.lock"
        self._fd = -1

    def __enter__(self) -> None:
        directory = os.path.dirname(self._lock_path)
        if directory:
            ensure_dir(directory)

        flags = os.O_RDWR | os.O_CREAT
        if WINDOWS:
            flags |= getattr(os, "O_BINARY", 0)
        self._fd = os.open(self._lock_path, flags, 0o600)
        try:
            if WINDOWS:
                import msvcrt

                # msvcrt.locking can only lock bytes that exist.
                if os.fstat(self._fd).st_size < 1:
                    os.write(self._fd, b"\n")
                os.lseek(self._fd, 0, os.SEEK_SET)
                # LK_LOCK retries for about 10 seconds before failing.
                msvcrt.locking(  # type: ignore[attr-defined]
                    self._fd, msvcrt.LK_LOCK, 1  # type: ignore[attr-defined]
                )
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            raise

    def __exit__(self, *exc_info: object) -> None:
        if self._fd < 0:
            return
        try:
            if WINDOWS:
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(  # type: ignore[attr-defined]
                    self._fd, msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
                )
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


class Configuration:
    """Handles management of configuration.

    Provides an interface to accessing and managing configuration files.

    This class converts provides an API that takes "section.key-name" style
    keys and stores the value associated with it as "key-name" under the
    section "section".

    This allows for a clean interface wherein the both the section and the
    key-name are preserved in an easy to manage form in the configuration files
    and the data stored is also nice.
    """

    def __init__(self, isolated: bool, load_only: Kind | None = None) -> None:
        super().__init__()

        if load_only is not None and load_only not in VALID_LOAD_ONLY:
            raise ConfigurationError(
                "Got invalid value for load_only - should be one of {}".format(
                    ", ".join(map(repr, VALID_LOAD_ONLY))
                )
            )
        self.isolated = isolated
        self.load_only = load_only

        # Because we keep track of where we got the data from
        self._parsers: dict[Kind, list[tuple[str, RawConfigParser]]] = {
            variant: [] for variant in OVERRIDE_ORDER
        }
        self._config: dict[Kind, dict[str, dict[str, Any]]] = {
            variant: {} for variant in OVERRIDE_ORDER
        }
        self._modified_parsers: list[tuple[str, RawConfigParser]] = []

        # Original bytes of each file at load time (b"" if missing) used
        # for conflict detection, and the set/unset edits requested since.
        self._snapshots: dict[str, bytes] = {}
        self._pending_edits: dict[str, list[_Edit]] = {}

    def load(self) -> None:
        """Loads configuration from configuration files and environment"""
        self._load_config_files()
        if not self.isolated:
            self._load_environment_vars()

    def get_file_to_edit(self) -> str | None:
        """Returns the file with highest priority in configuration"""
        assert self.load_only is not None, "Need to be specified a file to be editing"

        try:
            return self._get_parser_to_modify()[0]
        except IndexError:
            return None

    def items(self) -> Iterable[tuple[str, Any]]:
        """Returns key-value pairs like dict.items() representing the loaded
        configuration
        """
        return self._dictionary.items()

    def get_value(self, key: str) -> Any:
        """Get a value from the configuration."""
        orig_key = key
        key = _normalize_name(key)
        try:
            clean_config: dict[str, Any] = {}
            for file_values in self._dictionary.values():
                clean_config.update(file_values)
            return clean_config[key]
        except KeyError:
            # disassembling triggers a more useful error message than simply
            # "No such key" in the case that the key isn't in the form command.option
            _disassemble_key(key)
            raise ConfigurationError(f"No such key - {orig_key}")

    def set_value(self, key: str, value: Any) -> None:
        """Modify a value in the configuration."""
        key = _normalize_name(key)
        self._ensure_have_load_only()

        assert self.load_only
        fname, parser = self._get_parser_to_modify()
        section, name = _disassemble_key(key)

        if parser is not None:
            # Modify the parser and the configuration
            if not parser.has_section(section):
                parser.add_section(section)
            parser.set(section, name, value)

        self._config[self.load_only].setdefault(fname, {})
        self._config[self.load_only][fname][key] = value
        self._register_edit(fname, _Edit("set", section, name, str(value)))
        self._mark_as_modified(fname, parser)

    def unset_value(self, key: str) -> None:
        """Unset a value in the configuration."""
        orig_key = key
        key = _normalize_name(key)
        self._ensure_have_load_only()

        assert self.load_only
        fname, parser = self._get_parser_to_modify()

        if (
            key not in self._config[self.load_only][fname]
            and key not in self._config[self.load_only]
        ):
            raise ConfigurationError(f"No such key - {orig_key}")

        if parser is not None:
            section, name = _disassemble_key(key)
            if not (
                parser.has_section(section) and parser.remove_option(section, name)
            ):
                # The option was not removed.
                raise ConfigurationError(
                    "Fatal Internal error [id=1]. Please report as a bug."
                )

            # The section may be empty after the option was removed.
            if not parser.items(section):
                parser.remove_section(section)
            self._register_edit(fname, _Edit("unset", section, name))
            self._mark_as_modified(fname, parser)
        try:
            del self._config[self.load_only][fname][key]
        except KeyError:
            del self._config[self.load_only][key]

    def save(self) -> None:
        """Save the current in-memory state."""
        self._ensure_have_load_only()

        for fname, edits in self._pending_edits.items():
            # Ensure directory exists.
            directory = os.path.dirname(fname)
            if directory:
                ensure_dir(directory)

            # Ensure directory's permission(need to be writeable)
            try:
                self._save_file(fname, edits)
            except ConfigurationError:
                raise
            except (OSError, UnicodeError) as error:
                raise ConfigurationError(
                    f"An error occurred while writing to the configuration file "
                    f"{fname}: {error}"
                )

    def _save_file(self, fname: str, edits: list[_Edit]) -> None:
        """Verify, patch and atomically replace a single config file."""
        with _ExclusiveFileLock(fname):
            try:
                with open(fname, "rb") as f:
                    current_bytes = f.read()
                existing_stat = os.stat(fname)
            except FileNotFoundError:
                current_bytes = b""
                existing_stat = None

            # Conflict detection: the bytes on disk must still be the
            # bytes our edits were computed against.
            if current_bytes != self._snapshots.get(fname, b""):
                raise ConfigurationError(
                    f"Configuration file {fname} was modified by another "
                    "process after it was read, so the change was not "
                    "written. Re-run the command to apply it."
                )

            encoding = get_locale_encoding()
            new_text = _patch_config_text(current_bytes.decode(encoding), edits)
            new_bytes = new_text.encode(encoding)

            # Idempotency: nothing to do when the result is identical.
            if new_bytes == current_bytes:
                logger.debug("No changes needed in %s", fname)
                return

            logger.info("Writing to %s", fname)
            self._atomic_replace(fname, new_bytes, existing_stat)

    def _atomic_replace(
        self,
        fname: str,
        data: bytes,
        existing_stat: os.stat_result | None,
    ) -> None:
        """Write ``data`` next to ``fname`` and atomically rename it over."""
        directory = os.path.dirname(fname) or os.curdir
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{os.path.basename(fname)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                # Survive an abrupt process exit with no truncated file.
                os.fsync(f.fileno())
            if existing_stat is not None:
                os.chmod(tmp_name, existing_stat.st_mode & 0o7777)
            os.replace(tmp_name, fname)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        self._fsync_directory(directory)

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        """Best-effort durability of the rename itself (POSIX only)."""
        if WINDOWS:
            return
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    #
    # Private routines
    #

    def _ensure_have_load_only(self) -> None:
        if self.load_only is None:
            raise ConfigurationError("Needed a specific file to be modifying.")
        logger.debug("Will be working with %s variant only", self.load_only)

    @property
    def _dictionary(self) -> dict[str, dict[str, Any]]:
        """A dictionary representing the loaded configuration."""
        # NOTE: Dictionaries are not populated if not loaded. So, conditionals
        #       are not needed here.
        retval = {}

        for variant in OVERRIDE_ORDER:
            retval.update(self._config[variant])

        return retval

    def _load_config_files(self) -> None:
        """Loads configuration from configuration files"""
        config_files = dict(self.iter_config_files())
        if config_files[kinds.ENV][0:1] == [os.devnull]:
            logger.debug(
                "Skipping loading configuration files due to "
                "environment's PIP_CONFIG_FILE being os.devnull"
            )
            return

        for variant, files in config_files.items():
            for fname in files:
                # If there's specific variant set in `load_only`, load only
                # that variant, not the others.
                if self.load_only is not None and variant != self.load_only:
                    logger.debug("Skipping file '%s' (variant: %s)", fname, variant)
                    continue

                parser = self._load_file(variant, fname)

                # Keeping track of the parsers used
                self._parsers[variant].append((fname, parser))

    def _load_file(self, variant: Kind, fname: str) -> RawConfigParser:
        logger.verbose("For variant '%s', will try loading '%s'", variant, fname)
        parser = self._construct_parser(fname)

        for section in parser.sections():
            items = parser.items(section)
            self._config[variant].setdefault(fname, {})
            self._config[variant][fname].update(self._normalized_keys(section, items))

        return parser

    def _construct_parser(self, fname: str) -> RawConfigParser:
        parser = configparser.RawConfigParser()
        locale_encoding = get_locale_encoding()
        # If there is no such file, don't bother reading it but create the
        # parser anyway, to hold the data.
        # Doing this is useful when modifying and saving files, where we don't
        # need to construct a parser.
        if os.path.exists(fname):
            try:
                with open(fname, "rb") as f:
                    raw_bytes = f.read()
                text = raw_bytes.decode(locale_encoding)
            except UnicodeDecodeError:
                # See https://github.com/pypa/pip/issues/4963
                self._snapshots.pop(fname, None)
                raise ConfigurationFileCouldNotBeLoaded(
                    reason=f"contains invalid {locale_encoding} characters",
                    fname=fname,
                )
            try:
                parser.read_string(text, source=fname)
            except configparser.Error as error:
                # See https://github.com/pypa/pip/issues/4893
                self._snapshots.pop(fname, None)
                raise ConfigurationFileCouldNotBeLoaded(error=error)
            self._snapshots[fname] = raw_bytes
        else:
            self._snapshots[fname] = b""
        return parser

    def _load_environment_vars(self) -> None:
        """Loads configuration from environment variables"""
        self._config[kinds.ENV_VAR].setdefault(":env:", {})
        self._config[kinds.ENV_VAR][":env:"].update(
            self._normalized_keys(":env:", self.get_environ_vars())
        )

    def _normalized_keys(
        self, section: str, items: Iterable[tuple[str, Any]]
    ) -> dict[str, Any]:
        """Normalizes items to construct a dictionary with normalized keys.

        This routine is where the names become keys and are made the same
        regardless of source - configuration files or environment.
        """
        normalized = {}
        for name, val in items:
            key = section + "." + _normalize_name(name)
            normalized[key] = val
        return normalized

    def get_environ_vars(self) -> Iterable[tuple[str, str]]:
        """Returns a generator with all environmental vars with prefix PIP_"""
        for key, val in os.environ.items():
            if key.startswith("PIP_"):
                name = key[4:].lower()
                if name not in ENV_NAMES_IGNORED:
                    yield name, val

    # XXX: This is patched in the tests.
    def iter_config_files(self) -> Iterable[tuple[Kind, list[str]]]:
        """Yields variant and configuration files associated with it.

        This should be treated like items of a dictionary. The order
        here doesn't affect what gets overridden. That is controlled
        by OVERRIDE_ORDER. However this does control the order they are
        displayed to the user. It's probably most ergonomic to display
        things in the same order as OVERRIDE_ORDER
        """
        # SMELL: Move the conditions out of this function

        env_config_file = os.environ.get("PIP_CONFIG_FILE", None)
        config_files = get_configuration_files()

        yield kinds.GLOBAL, config_files[kinds.GLOBAL]

        # per-user config is not loaded when env_config_file exists
        should_load_user_config = not self.isolated and not (
            env_config_file and os.path.exists(env_config_file)
        )
        if should_load_user_config:
            # The legacy config file is overridden by the new config file
            yield kinds.USER, config_files[kinds.USER]

        # virtualenv config
        yield kinds.SITE, config_files[kinds.SITE]

        if env_config_file is not None:
            yield kinds.ENV, [env_config_file]
        else:
            yield kinds.ENV, []

    def get_values_in_config(self, variant: Kind) -> dict[str, Any]:
        """Get values present in a config file"""
        return self._config[variant]

    def _get_parser_to_modify(self) -> tuple[str, RawConfigParser]:
        # Determine which parser to modify
        assert self.load_only
        parsers = self._parsers[self.load_only]
        if not parsers:
            # This should not happen if everything works correctly.
            raise ConfigurationError(
                "Fatal Internal error [id=2]. Please report as a bug."
            )

        # Use the highest priority parser.
        return parsers[-1]

    # XXX: This is patched in the tests.
    def _mark_as_modified(self, fname: str, parser: RawConfigParser) -> None:
        file_parser_tuple = (fname, parser)
        if file_parser_tuple not in self._modified_parsers:
            self._modified_parsers.append(file_parser_tuple)

    def _register_edit(self, fname: str, edit: _Edit) -> None:
        self._pending_edits.setdefault(fname, []).append(edit)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dictionary!r})"
